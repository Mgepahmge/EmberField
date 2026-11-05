#!/usr/bin/env python3
"""串行执行完整算例benchmark，分离预热与测量并保存输入、程序和阶段计时。"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def writeJson(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def revision(directory):
    """记录工作树版本；不能识别仓库时保留未知，不推测程序编译版本。"""
    result = subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def run(args):
    binary, source = args.executable.resolve(), args.case.resolve()
    limits = args.resourceLimits.resolve()
    target = args.output.resolve()
    target.mkdir(parents=True, exist_ok=False)
    definition = json.loads(source.read_text(encoding="utf-8"))
    mesh = (source.parent / definition["mesh"]["path"]).resolve()
    cache = binary.parent / "CMakeCache.txt"
    cacheText = cache.read_text(encoding="utf-8") if cache.exists() else ""
    buildFlags = [line for line in cacheText.splitlines() if line.startswith((
        "CMAKE_BUILD_TYPE:", "CMAKE_CUDA_ARCHITECTURES:", "EMBERFIELD_ENABLE_NVTX:"))]
    if not args.allowNvtx and any(line.endswith("=ON") and line.startswith("EMBERFIELD_ENABLE_NVTX:") for line in buildFlags):
        raise ValueError("正式计时须使用NVTX关闭构建；分析采集请显式传--allow-nvtx并单独命名结果")
    report = dict(schemaVersion=1, status="running", executable=str(binary), executableSha256=sha256(binary),
                  sourceTreeRevision=revision(source.parent), buildFlags=buildFlags,
                  host=dict(platform=platform.platform(), cpu=platform.processor(), logicalCpus=os.cpu_count(),
                            memoryInfo=Path("/proc/meminfo").read_text() if Path("/proc/meminfo").exists() else None),
                  input=dict(case=str(source), caseSha256=sha256(source), mesh=str(mesh), meshSha256=sha256(mesh),
                             limits=str(limits), limitsSha256=sha256(limits)),
                  warmupRuns=args.warmup, measuredRuns=args.repeat, runs=[])
    writeJson(target / "benchmark.json", report)
    try:
        for index in range(args.warmup + args.repeat):
            warmup = index < args.warmup
            name = f'warmup-{index + 1:02}' if warmup else f'run-{index - args.warmup + 1:02}'
            command = [str(binary), str(source), "--resource-limits", str(limits),
                       "--output-root", str(target), "--output", name]
            rss = target / (name + ".resource.txt")
            if Path("/usr/bin/time").exists():
                command = ["/usr/bin/time", "-v", "-o", str(rss), *command]
            started = time.perf_counter()
            with (target / (name + ".log")).open("w", encoding="utf-8") as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                           env={**os.environ, "LC_ALL": "C"})
            entry = dict(name=name, warmup=warmup, exitCode=completed.returncode,
                         processWallS=time.perf_counter() - started, command=command)
            report["runs"].append(entry)
            if rss.exists():
                for line in rss.read_text(encoding="utf-8").splitlines():
                    if "Maximum resident set size (kbytes):" in line:
                        entry["maximumResidentKiB"] = int(line.rsplit(":", 1)[1].strip())
            summaryFile = target / name / "run-summary.json"
            if summaryFile.exists():
                summary = json.loads(summaryFile.read_text(encoding="utf-8"))
                entry.update(status=summary["status"], mesh=summary["mesh"], timingS=summary["timingS"],
                             completedSteps=summary["completedSteps"])
                with (target / name / "steps.csv").open(newline="", encoding="utf-8") as stream:
                    rows = list(csv.DictReader(stream))
                accepted = [row for row in rows if row["kind"] == "accepted"]
                entry["iterations"] = sum(int(row["iterations"]) for row in accepted)
                entry["maximumAcceptedResidual"] = max((float(row["accepted_residual_l2"]) for row in accepted), default=None)
                entry["balanceStatuses"] = sorted({row["balance_status"] for row in accepted})
            writeJson(target / "benchmark.json", report)
            if completed.returncode or entry.get("status") != "completed":
                raise ValueError(f"{name}未完整成功；保留日志和失败状态，不计入benchmark统计")
            print(f'{name}: wall={entry["processWallS"]:.3f}s, timing={entry["timingS"]}', flush=True)
        measured = [r for r in report["runs"] if not r["warmup"]]
        # 相同输入必须保持相同问题规模和步数。
        if not all(r["mesh"] == measured[0]["mesh"] and r["completedSteps"] == measured[0]["completedSteps"] for r in measured):
            raise ValueError("重复运行的问题规模或实际步数不一致")
        metrics = {"processWallS": [r["processWallS"] for r in measured]}
        for name in measured[0]["timingS"]:
            metrics[name] = [r["timingS"][name] for r in measured]
        report["statistics"] = {name: dict(samples=values, median=statistics.median(values),
                                            minimum=min(values), maximum=max(values),
                                            mean=statistics.mean(values),
                                            standardDeviation=statistics.stdev(values) if len(values) > 1 else None)
                                for name, values in metrics.items()}
        report["status"] = "completed"
        report["statisticalBenchmark"] = args.repeat >= 3 and args.warmup >= 1
        writeJson(target / "benchmark.json", report)
    except BaseException as error:
        report.update(status="failed", error=str(error))
        writeJson(target / "benchmark.json", report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("case", type=Path)
    parser.add_argument("--resource-limits", dest="resourceLimits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--allow-nvtx", dest="allowNvtx", action="store_true")
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat < 1:
        parser.error("预热次数须非负，测量次数须为正")
    try:
        run(args)
    except (ValueError, OSError, KeyError) as error:
        print(f"benchmark失败：{error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
