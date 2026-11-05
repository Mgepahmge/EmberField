#!/usr/bin/env python3
"""一致细化现有Triangle3/Tet4原生算例，生成独立可复现的大规模输入包。"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys


def sha256(path):
    """流式计算文件哈希，避免将大网格整体读入内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def writeJson(path, value):
    """写入UTF-8 JSON元数据。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def inventory(gmsh, dimension):
    """统计网格、物理组与材料接口；返回的节点数包含所有保存节点。"""
    import numpy as np
    tags = gmsh.model.mesh.getNodes()[0]
    counts = {}
    domainTags = []
    for dim in range(dimension + 1):
        types, elementTags, nodes = gmsh.model.mesh.getElements(dim)
        for kind, items, connectivity in zip(types, elementTags, nodes):
            if dim == dimension:
                expected = 2 if dimension == 2 else 4
                if kind != expected:
                    raise ValueError("活动维度仅接受Triangle3或Tet4，不自动降阶")
                domainTags.append(connectivity)
            counts[str(int(kind))] = counts.get(str(int(kind)), 0) + len(items)
    active = np.unique(np.concatenate(domainTags)) if domainTags else np.array([], dtype=np.uint64)
    if not len(active):
        raise ValueError("网格活动维度没有单元")
    if len(active) != len(tags):
        raise ValueError("准备器要求所有保存节点属于活动维度；请先提供已提取活动域的网格")
    groups = []
    for dim, tag in gmsh.model.getPhysicalGroups():
        entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
        size = sum(sum(len(x) for x in gmsh.model.mesh.getElements(dim, int(e))[1]) for e in entities)
        groups.append(dict(dimension=dim, tag=tag, name=gmsh.model.getPhysicalName(dim, tag), elements=size))
    return dict(nodes=len(tags), activeDofs=len(active), elementTypes=counts, physicalGroups=groups)


def limitsFor(stats, dimension):
    """按网格规模设置显式阶段预算；预算不是进程内存限额。"""
    nodes = stats["nodes"]
    cells = stats["elementTypes"][str(2 if dimension == 2 else 4)]
    local = dimension + 1
    return {
        "schemaVersion": 1,
        "case": {"maxValues": max(1000000, cells * 12)},
        "mesh": {"maxRecords": max(100000000, cells * 32), "maxBytes": max(2147483648, cells * 128)},
        "space": {"maxCells": cells * 2, "maxLocalDofs": cells * local * 2, "maxGlobalDofs": nodes * 2},
        "preparation": {"maxCells": cells * 2, "maxDofs": nodes * 2, "maxFacetUses": cells * local * 2,
                        "maxAssociations": cells * 32},
        "assembly": {"maxDofs": nodes * 2, "maxStencils": cells * 2,
                     "maxLocalEntries": cells * local * local * 2, "maxNonzeros": cells * local * local * 2},
        "output": {"maxPoints": nodes * 2, "maxCells": cells * 2,
                   "maxHostBytes": max(4294967296, nodes * 256 + cells * 1600),
                   "maxDeviceBytes": max(2147483648, nodes * 192 + cells * 1152 + 16777216),
                   "maxOutputBytes": 107374182400},
    }


def prepare(args):
    """仅改变网格密度和明确请求的benchmark设置；原始输入随包保留。"""
    import gmsh
    source = args.case.resolve()
    definition = json.loads(source.read_text(encoding="utf-8"))
    mesh = (source.parent / definition["mesh"]["path"]).resolve()
    dimension = {"planar2d": 2, "solid3d": 3}[definition["analysis"]["space"]]
    destination = args.output.resolve()
    if destination.exists():
        raise ValueError("输出目录已存在，禁止覆盖benchmark输入")
    destination.mkdir(parents=True)
    shutil.copy2(source, destination / "source-case.json")
    sourceHash, meshHash = sha256(source), sha256(mesh)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.NumThreads", args.threads)
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.open(str(mesh))
        original = inventory(gmsh, dimension)
        originalGroups = {(x["dimension"], x["tag"]): x for x in original["physicalGroups"]}
        for depth in range(max(args.levels) + 1):
            reusedMesh = None
            if args.reuseFrom:
                if depth not in args.levels:
                    continue
                previous = args.reuseFrom.resolve() / f"level-{depth}"
                manifest = json.loads((previous / "manifest.json").read_text(encoding="utf-8"))
                reusedMesh = previous / "mesh.msh"
                if (manifest["sourceMeshSha256"] != meshHash or manifest["refinementLevel"] != depth
                        or manifest["meshSha256"] != sha256(reusedMesh)):
                    raise ValueError("复用网格与原始来源或细化层数不一致")
                gmsh.clear()
                gmsh.open(str(reusedMesh))
            elif depth:
                gmsh.model.mesh.refine()
            if depth not in args.levels:
                continue
            stats = inventory(gmsh, dimension)
            current = {(x["dimension"], x["tag"]): x for x in stats["physicalGroups"]}
            if current.keys() != originalGroups.keys():
                raise ValueError("细化改变了物理组集合")
            for key, initial in originalGroups.items():
                if current[key]["name"] != initial["name"] or current[key]["elements"] != initial["elements"] * (2 ** initial["dimension"]) ** depth:
                    raise ValueError("细化未按预期保留物理组名称或单元覆盖")
            directory = destination / f"level-{depth}"
            directory.mkdir()
            gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
            gmsh.option.setNumber("Mesh.Binary", 1)
            gmsh.option.setNumber("Mesh.SaveAll", 1)
            if reusedMesh:
                shutil.copy2(reusedMesh, directory / "mesh.msh")
            else:
                gmsh.write(str(directory / "mesh.msh"))
            case = json.loads(json.dumps(definition))
            case["name"] = f'{definition.get("name", source.stem)} / refined-{depth}'
            case["mesh"]["path"] = "mesh.msh"
            if args.steps is not None:
                if case["analysis"]["type"] != "transient":
                    raise ValueError("--steps仅用于瞬态，不自动补造材料热容或初态")
                case["time"]["step"] = (case["time"]["end"] - case["time"]["start"]) / args.steps
                case["time"]["maxSteps"] = max(case["time"].get("maxSteps", 1), args.steps * 4)
            if args.relativeTolerance is not None:
                case.setdefault("linearSolver", {})["relativeTolerance"] = args.relativeTolerance
            if args.absoluteTolerance is not None:
                case.setdefault("linearSolver", {})["absoluteTolerance"] = args.absoluteTolerance
            # 只保存末帧，不让输出频率随网格或实际事件数改变。
            output = case.setdefault("output", {})
            output.update(format="vtu", encoding="appendedRaw", overwrite=False,
                          fields=["temperature"], includeInitial=False, includeFinal=True)
            if case["analysis"]["type"] == "transient":
                output["schedule"] = {"kind": "everySteps", "interval": 1000000000}
            else:
                output.pop("schedule", None)
                output.pop("includeInitial", None)
            output.pop("directory", None)
            if args.withoutMonitors:
                output["monitors"] = []
            writeJson(directory / "case.json", case)
            writeJson(directory / "limits.json", limitsFor(stats, dimension))
            report = dict(schemaVersion=1, gmshVersion=gmsh.__version__, refinementLevel=depth,
                          sourceCase=str(source), sourceCaseSha256=sourceHash,
                          sourceMesh=str(mesh), sourceMeshSha256=meshHash,
                          meshSha256=sha256(directory / "mesh.msh"), caseSha256=sha256(directory / "case.json"),
                          limitsSha256=sha256(directory / "limits.json"), mesh=stats,
                          reusedMesh=str(reusedMesh) if reusedMesh else None,
                          changes=dict(nominalSteps=args.steps, relativeTolerance=args.relativeTolerance,
                                       absoluteTolerance=args.absoluteTolerance,
                                       withoutMonitors=args.withoutMonitors,
                                       output="temperature/final-frame-only"),
                          geometry="Uniform straight-sided refinement of the existing discrete mesh; no CAD projection.")
            writeJson(directory / "manifest.json", report)
            print(json.dumps(dict(directory=str(directory), nodes=stats["nodes"], elements=stats["elementTypes"])), flush=True)
    finally:
        gmsh.finalize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--relative-tolerance", dest="relativeTolerance", type=float)
    parser.add_argument("--absolute-tolerance", dest="absoluteTolerance", type=float)
    parser.add_argument("--without-monitors", dest="withoutMonitors", action="store_true", help="显式关闭可选点/区域监测，保留热平衡与最终温度场")
    parser.add_argument("--reuse-from", dest="reuseFrom", type=Path, help="复用同一原始网格已生成的level-N网格，保证不同物理配置的节点和单元顺序相同")
    args = parser.parse_args()
    if min(args.levels) < 0 or args.threads < 1 or (args.steps is not None and args.steps < 1):
        parser.error("细化层数须非负，线程和步数须为正")
    if args.relativeTolerance is not None and not 0 < args.relativeTolerance < 1:
        parser.error("相对容差须位于(0,1)")
    if args.absoluteTolerance is not None and not 0 <= args.absoluteTolerance < float("inf"):
        parser.error("绝对容差须为有限非负数")
    try:
        prepare(args)
    except (ValueError, OSError, KeyError, ImportError) as error:
        print(f"准备失败：{error}；部分生成目录保留用于检查，不作为完整输入包。", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
