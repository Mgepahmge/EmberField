"""Black-box CLI and published-file checks; --vtk additionally uses an independent VTK reader."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET


def run(command, cwd, expected=0):
    result = subprocess.run([str(x) for x in command], cwd=cwd, capture_output=True, text=True)
    assert (result.returncode == expected if expected is not None else result.returncode != 0), (
        result.returncode, result.stdout, result.stderr
    )
    return result


def check_directory(directory, transient, vtk_enabled):
    summary = json.loads((directory / "run-summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == summary["solveStatus"] == summary["outputStatus"] == "completed"
    assert summary["lastAcceptedStep"] == summary["lastPersistedStep"]
    assert summary["input"]["effectiveCase"]["output"]["directory"] == str(directory)
    for table in summary["tables"].values():
        if table is None:
            continue
        data = (directory / table["path"]).read_bytes()
        assert len(data) == int(table["committedBytes"])
        assert len(list(csv.reader(data.decode("utf-8").splitlines()))) == int(table["committedRows"]) + 1
    frames = summary["frames"]
    assert len({f["path"] for f in frames}) == len(frames)
    for frame in frames:
        path = directory / frame["path"]
        assert path.stat().st_size == int(frame["bytes"])
        if vtk_enabled:
            import vtk
            reader = vtk.vtkXMLUnstructuredGridReader()
            errors = []
            reader.AddObserver("ErrorEvent", lambda *_: errors.append(True))
            reader.SetFileName(str(path))
            reader.Update()
            grid = reader.GetOutput()
            assert not errors and grid.GetNumberOfPoints() == int(summary["mesh"]["pointCount"])
            assert grid.GetNumberOfCells() == int(summary["mesh"]["cellCount"])
            point = grid.GetPointData()
            assert point.GetArray("temperature").GetNumberOfTuples() == grid.GetNumberOfPoints()
            assert point.GetArray("sourceNodeTag").GetDataTypeAsString() == "unsigned long long"
            for definition in summary["fieldDefinitions"]:
                data = point if definition["association"] == "point" else grid.GetCellData()
                array = data.GetArray(definition["name"])
                assert array and array.GetNumberOfComponents() == definition["components"]
                assert all(math.isfinite(x) for i in range(array.GetNumberOfTuples()) for x in array.GetTuple(i))
            assert grid.GetFieldData().GetArray("stepIndex").GetValue(0) == int(frame["step"])
            if transient:
                assert grid.GetFieldData().GetArray("TimeValue").GetValue(0) == frame["timeS"]
    pvd = directory / "temperature.pvd"
    assert pvd.exists() == (transient and bool(frames))
    if pvd.exists():
        entries = ET.parse(pvd).findall("./Collection/DataSet")
        assert [(e.attrib["file"], float(e.attrib["timestep"])) for e in entries] == [
            (f["path"], f["timeS"]) for f in frames
        ]
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path)
    parser.add_argument("cases", type=Path)
    parser.add_argument("--vtk", action="store_true")
    args = parser.parse_args()
    source = json.loads((args.cases / "steady.json").read_text(encoding="utf-8"))
    source["mesh"]["path"] = str((args.cases / "plate.msh").resolve())
    source["output"] = {"fields": ["temperature", "temperatureGradient", "heatFlux"], "encoding": "ascii"}
    with tempfile.TemporaryDirectory(prefix="emberfield-cli-") as folder:
        root = Path(folder)
        binary = root / "bin" / args.executable.name
        binary.parent.mkdir()
        shutil.copy2(args.executable, binary)
        run([binary, "--help"], root)
        run([binary, "--version"], root)
        run([binary], root, expected=None)
        case = root / "plate.multi.json"
        case.write_text(json.dumps(source), encoding="utf-8")
        run([binary, case], root)
        default = check_directory(binary.parent / "plate.multi", False, args.vtk)
        assert default["frames"][0]["timeS"] is None
        run([binary, case], root, expected=None)
        run([binary, case, "--output", "../escape"], root, expected=None)
        source["analysis"]["type"] = "transient"
        source["initialState"] = {"defaultTemperature": {"kind": "constant", "value": 300}}
        source["time"] = {"start": 0, "end": .25, "step": .1}
        for material in source["materials"]:
            material["volumetricHeatCapacity"] = {"kind": "constant", "value": 3900000}
        source["output"].update({"encoding": "appendedRaw", "includeInitial": False, "includeFinal": True,
                                  "schedule": {"kind": "everySteps", "interval": 2}})
        case.write_text(json.dumps(source), encoding="utf-8")
        run([binary, case, "--output-root", root / "results", "--output", "transient"], root)
        transient = check_directory(root / "results/transient", True, args.vtk)
        assert [int(f["step"]) for f in transient["frames"]] == [2, 3]
        with (root / "results/transient/steps.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 4 and rows[0]["kind"] == "initial"
        assert math.isclose(float(rows[-1]["dt_s"]), .05) and float(rows[-1]["time_s"]) == .25
        source["output"]["includeFinal"] = False
        case.write_text(json.dumps(source), encoding="utf-8")
        run([binary, case, "--output-root", root / "results", "--output", "interval-only"], root)
        interval = check_directory(root / "results/interval-only", True, args.vtk)
        assert [int(f["step"]) for f in interval["frames"]] == [2]
        source["output"]["schedule"]["interval"] = 100
        case.write_text(json.dumps(source), encoding="utf-8")
        run([binary, case, "--output-root", root / "results", "--output", "no-fields"], root)
        no_fields = check_directory(root / "results/no-fields", True, args.vtk)
        assert not no_fields["frames"]
        source["linearSolver"] = {"method": "cg", "maxIterations": 1, "relativeTolerance": 1e-14,
                                  "absoluteTolerance": 1e-14}
        case.write_text(json.dumps(source), encoding="utf-8")
        run([binary, case, "--output-root", root / "results", "--output", "failed"], root, expected=3)
        failure = json.loads((root / "results/failed/run-summary.json").read_text(encoding="utf-8"))
        assert failure["status"] == failure["solveStatus"] == "failed"
        assert not failure["frames"] and failure["lastAcceptedStep"] == "0"
        assert failure["lastPersistedStep"] == "0"
        if os.name == "posix":
            source["linearSolver"] = {"method": "cg", "maxIterations": 1000,
                                      "relativeTolerance": 1e-8, "absoluteTolerance": 1e-10}
            source["time"] = {"start": 0, "end": 100, "step": .001}
            case.write_text(json.dumps(source), encoding="utf-8")
            directory = root / "results/interrupted"
            process = subprocess.Popen([str(binary), str(case), "--output-root", str(root / "results"),
                                        "--output", "interrupted"], cwd=root,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            try:
                deadline = time.monotonic() + 15
                while not (directory / "steps.csv").exists():
                    assert process.poll() is None, process.stderr.read()
                    assert time.monotonic() < deadline, "output session did not start"
                    time.sleep(.01)
                process.terminate()
                _, errors = process.communicate(timeout=15)
                assert process.returncode == 130, (process.returncode, errors)
                interrupted = json.loads((directory / "run-summary.json").read_text(encoding="utf-8"))
                assert interrupted["status"] == interrupted["solveStatus"] == "interrupted"
                assert interrupted["lastAcceptedStep"] == interrupted["lastPersistedStep"]
                assert not Path(str(directory) + ".emberfield-lock").exists()
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate()
    print("CLI defaults, strict paths, ASCII/raw VTU, actual dt, scheduling, table prefixes and numerical failure: passed")


if __name__ == "__main__":
    main()
