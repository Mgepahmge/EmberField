"""转换器共用的中立模型、严格检查与事务发布；只依赖Python标准库。"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile


class ConversionError(ValueError):
    """源输入无法无歧义转换时的错误，不发布部分算例。"""


def require(condition, message):
    """验证输入条件，失败抛出带上下文的转换错误。"""
    if not condition:
        raise ConversionError(message)


def number(text):
    """读取有限实数，兼容Fortran D指数，不执行表达式。"""
    try:
        value = float(str(text).replace("D", "e").replace("d", "e"))
    except ValueError as exc:
        raise ConversionError(f"不是常数实数: {text!r}") from exc
    require(math.isfinite(value), f"数值必须有限: {text!r}")
    return value


def integer(text):
    """精确读取正整数身份，不经过浮点转换。"""
    try:
        value = int(text)
    except ValueError as exc:
        raise ConversionError(f"不是整数字面量: {text!r}") from exc
    require(0 < value <= 2**63 - 1, f"整数须位于1..INT64_MAX: {text}")
    return value


def constant(value):
    """构造EmberField常量标量字段。"""
    return {"kind": "constant", "value": value}


def stepLimit(duration, step):
    """验证步数计算有限且可由目标int64时间预算表达，返回含末步余量的上限。"""
    require(math.isfinite(duration) and duration > 0 and math.isfinite(step) and step > 0,
            "时间范围和时间步必须正且有限")
    ratio = duration / step
    require(math.isfinite(ratio) and ratio < 2**63 - 2, "时间步数超过int64范围")
    return math.ceil(ratio) + 1


def kelvin(value, unit):
    """按调用者显式声明将源温度转换为绝对温度K。"""
    result = value + (273.15 if unit == "C" else 0)
    require(math.isfinite(result) and result >= 0, f"绝对温度非法: {result} K")
    return result


@dataclass
class Model:
    """拥有SI线性标量热模型；单元、区域及边界与源文件语法分离。"""
    dimension: int
    nodes: dict[int, tuple[float, float, float]] = field(default_factory=dict)
    cells: dict[int, tuple[int, ...]] = field(default_factory=dict)
    materials: dict[str, dict] = field(default_factory=dict)
    assignments: dict[str, set[int]] = field(default_factory=dict)
    sources: list[tuple[set[int], float]] = field(default_factory=list)
    boundaries: list[tuple[set[tuple[int, ...]], dict]] = field(default_factory=list)
    initial: dict[int, float] = field(default_factory=dict)
    time: dict | None = None
    thickness: float | None = None
    notes: list[str] = field(default_factory=list)
    inputs: set[Path] = field(default_factory=set)

    def facets(self):
        """建立完整侧到母单元映射；非流形拓扑立即拒绝。"""
        result = defaultdict(list)
        for tag, nodes in self.cells.items():
            for i in range(len(nodes)):
                face = tuple(sorted(nodes[:i] + nodes[i + 1:]))
                result[face].append(tag)
                require(len(result[face]) <= 2, f"非流形侧: {face}")
        return result

    def validate(self):
        """发布前验证拓扑、材料覆盖、可表达边界和瞬态状态。"""
        require(self.dimension in (2, 3) and self.nodes and self.cells, "需要非空二维或三维单纯形网格")
        if self.dimension == 2:
            require(self.thickness is not None and math.isfinite(self.thickness) and self.thickness > 0, "二维算例必须明确正厚度[m]")
        if self.time:
            require(0 <= self.time['start'] < self.time['end'] and math.isfinite(self.time['end'])
                    and math.isfinite(self.time['step']) and self.time['step'] > 0
                    and 0 < self.time['maxSteps'] <= 2**63-1, "时间配置越界或非法")
        seen = set()
        geometric = set()
        for tag, nodes in self.cells.items():
            require(len(nodes) == self.dimension + 1 and len(set(nodes)) == len(nodes), f"单元{tag}不是线性单纯形")
            require(set(nodes) <= self.nodes.keys(), f"单元{tag}引用缺失节点")
            key = tuple(sorted(nodes))
            require(key not in geometric, f"几何重复单元: {tag}")
            geometric.add(key)
            p = [self.nodes[n] for n in nodes]
            if self.dimension == 2:
                require(all(abs(x[2]) <= 1e-12 for x in p), "二维网格必须位于全局XY平面")
                det = (p[1][0]-p[0][0])*(p[2][1]-p[0][1]) - (p[2][0]-p[0][0])*(p[1][1]-p[0][1])
            else:
                a, b, c = [[p[j][i]-p[0][i] for i in range(3)] for j in (1, 2, 3)]
                det = sum(a[i]*(b[(i+1)%3]*c[(i+2)%3]-b[(i+2)%3]*c[(i+1)%3]) for i in range(3))
            require(math.isfinite(det) and det != 0, f"退化或溢出单元: {tag}")
            if det < 0:
                self.cells[tag] = (nodes[1], nodes[0], *nodes[2:])
        for name, members in self.assignments.items():
            require(name in self.materials and members and members <= self.cells.keys(), f"材料分配非法: {name}")
            require(not seen.intersection(members), f"材料覆盖重复: {name}")
            seen.update(members)
            material = self.materials[name]
            for key in ("conductivity", "density", "specificHeat", "volumetricHeatCapacity"):
                if key in material:
                    require(math.isfinite(material[key]) and material[key] > 0, f"材料{name}的{key}必须正且有限")
            require("conductivity" in material, f"材料{name}缺少导热率")
            if self.time:
                require("volumetricHeatCapacity" in material or {"density", "specificHeat"} <= material.keys(), f"瞬态材料{name}缺少热容量")
            if {'density', 'specificHeat'} <= material.keys():
                require(math.isfinite(material['density']*material['specificHeat']), "体积热容量溢出")
        require(seen == self.cells.keys(), "材料必须恰好覆盖所有活动单元")
        for members, source in self.sources:
            require(members and members <= self.cells.keys() and math.isfinite(source), "体源区域或数值非法")
        facets = self.facets()
        conditions = defaultdict(list)
        nodeTemperatures = {}
        for faces, condition in self.boundaries:
            require(faces, "不允许空边界")
            for face in faces:
                require(face in facets and len(facets[face]) == 1, f"边界不是完整外侧: {face}")
                conditions[face].append(condition["kind"])
                if condition["kind"] == "temperature":
                    for node in face:
                        t = condition["temperature"]["value"]
                        require(node not in nodeTemperatures or nodeTemperatures[node] == t, f"节点{node}温度约束冲突")
                        nodeTemperatures[node] = t
        for face, kinds in conditions.items():
            require("temperature" not in kinds or all(k == "temperature" for k in kinds), f"同侧温度与自然条件冲突: {face}")
        if self.time:
            require(self.initial.keys() == self.cells.keys(), "瞬态初温必须覆盖全部单元")
            for cell, value in self.initial.items():
                require(math.isfinite(value) and value >= 0, "初始绝对温度必须非负有限")
                for node in self.cells[cell]:
                    require(node not in nodeTemperatures or nodeTemperatures[node] == value,
                            f"初温与节点{node}规定温度不相容；本工具拒绝初始跳变")


def writeBundle(model, output, sourceFormat, sourceName):
    """验证中立模型，写出共享物理组的MSH4.1和配置，最后原子重命名新目录。

    输出目录必须不存在；失败移除自身临时目录。报告记录输入哈希与明确的数值设置。
    """
    model.validate()
    output = Path(output).absolute()
    require(not output.exists(), f"输出已存在，不覆盖: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    groups = []

    def group(dimension, members, name):
        require(members, f"空区域: {name}")
        tag = len(groups) + 1
        groups.append((dimension, tag, set(members), name))
        return {"kind": "physicalGroup", "dimension": dimension, "tag": tag}

    domain = group(model.dimension, model.cells, "domain")
    materials, assignments, sources, boundaries = [], [], [], []
    for i, (name, members) in enumerate(model.assignments.items(), 1):
        materialId = f"material-{i}"
        materials.append({"id": materialId, "displayName": name,
                          **{k: constant(v) for k, v in model.materials[name].items()}})
        assignments.append({"id": f"assignment-{i}", "material": materialId,
                            "region": group(model.dimension, members, f"material-{i}")})
    for i, (members, value) in enumerate(model.sources, 1):
        sources.append({"id": f"source-{i}", "region": group(model.dimension, members, f"source-{i}"),
                        "kind": "volumetric", "value": constant(value)})
    for i, (faces, condition) in enumerate(model.boundaries, 1):
        boundaries.append({"id": f"boundary-{i}", "region": group(model.dimension-1, faces, f"boundary-{i}"), **condition})
    definition = {"schemaVersion": 1, "name": sourceName,
                  "mesh": {"path": "mesh.msh", "format": "msh", "coordinateScale": 1},
                  "analysis": {"type": "transient" if model.time else "steady",
                               "space": "planar2d" if model.dimension == 2 else "solid3d", "domain": domain},
                  "materials": materials, "materialAssignments": assignments, "sources": sources,
                  "boundaries": boundaries, "boundaryPolicy": {"defaultBoundary": "adiabatic", "naturalOverlap": "add"},
                  "discretization": {"family": "lagrangeContinuous", "order": 1, "quadrature": "auto"},
                  "linearSolver": {"method": "cg", "preconditioner": {"kind": "identity"},
                                   "relativeTolerance": 1e-10, "absoluteTolerance": 1e-12, "maxIterations": 10000,
                                   "initialGuess": "previousStep" if model.time else "zero"},
                  "output": {"format": "vtu", "fields": ["temperature"], "overwrite": False}}
    if model.dimension == 2:
        definition["analysis"]["thickness"] = model.thickness
    if model.time:
        definition["time"] = model.time
        values = defaultdict(set)
        for cell, value in model.initial.items():
            values[value].add(cell)
        default = max(values, key=lambda v: len(values[v]))
        definition["initialState"] = {"defaultTemperature": constant(default), "overrides": [
            {"id": f"initial-{i}", "region": group(model.dimension, cells, f"initial-{i}"), "temperature": constant(value)}
            for i, (value, cells) in enumerate(values.items(), 1) if value != default]}
    # 一个实体对应完全相同的物理组签名，保留交叠分组而不复制单元。
    records = [(model.dimension, tag, nodes) for tag, nodes in model.cells.items()]
    faceSet = set().union(*(faces for faces, _ in model.boundaries)) if model.boundaries else set()
    nextTag = max(model.cells) + 1
    require(nextTag + len(faceSet) <= 2**63-1, "生成边界单元tag溢出")
    records += [(model.dimension-1, nextTag+i, face) for i, face in enumerate(sorted(faceSet))]
    entities = {}
    blocks = defaultdict(list)
    for dim, tag, nodes in records:
        member = tag if dim == model.dimension else tuple(sorted(nodes))
        signature = tuple(g for d, g, members, _ in groups if d == dim and member in members)
        key = (dim, signature)
        if key not in entities:
            entities[key] = len(entities) + 1
        blocks[key].append((tag, nodes))
    xyz = list(model.nodes.values())
    bbox = [min(p[i] for p in xyz) for i in range(3)] + [max(p[i] for p in xyz) for i in range(3)]
    lines = ["$MeshFormat", "4.1 0 8", "$EndMeshFormat", "$PhysicalNames", str(len(groups))]
    lines += [f'{d} {g} "{name}"' for d, g, _, name in groups]
    lines += ["$EndPhysicalNames", "$Entities", " ".join(str(sum(k[0] == d for k in entities)) for d in range(4))]
    for (dim, signature), entity in sorted(entities.items()):
        lines.append(" ".join(map(str, [entity, *bbox, len(signature), *signature, 0])))
    ownerEntity = next(entity for (dim, _), entity in entities.items() if dim == model.dimension)
    lines += ["$EndEntities", "$Nodes", f"1 {len(model.nodes)} {min(model.nodes)} {max(model.nodes)}",
              f"{model.dimension} {ownerEntity} 0 {len(model.nodes)}"]
    lines += [str(n) for n in model.nodes]
    lines += [" ".join(format(x, ".17g") for x in p) for p in model.nodes.values()]
    lines += ["$EndNodes", "$Elements", f"{len(blocks)} {len(records)} {min(r[1] for r in records)} {max(r[1] for r in records)}"]
    for key, block in blocks.items():
        dim = key[0]
        elementType = {1: 1, 2: 2, 3: 4}[dim]
        lines.append(f"{dim} {entities[key]} {elementType} {len(block)}")
        lines += [" ".join(map(str, (tag, *nodes))) for tag, nodes in block]
    lines += ["$EndElements", ""]
    report = {"schemaVersion": 1, "sourceFormat": sourceFormat, "sourceName": sourceName,
              "units": "SI; temperature K", "dimension": model.dimension,
              "nodeCount": len(model.nodes), "cellCount": len(model.cells),
              "notes": model.notes + ["数值求解设置采用EmberField CG+恒等预条件，不复制源程序求解器。",
                                        "输出使用EmberField默认VTU温度场，不复制源程序输出指令。"],
              "inputs": [{"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                         for p in sorted(model.inputs)], "case": "case.json", "mesh": "mesh.msh"}
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        (temporary / "mesh.msh").write_text("\n".join(lines), encoding="utf-8")
        for name, value in (("case.json", definition), ("conversion-report.json", report)):
            (temporary / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output
