"""Elmer SIF常量热传导适配；不执行MATC、用户过程或Include命令。"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import math
import re
import shlex

from .common import Model, constant, integer, kelvin, number, require, stepLimit


@dataclass
class Section:
    """拥有一个SIF节的字段，保留已消费状态以拒绝遗漏物理设置。"""
    kind: str
    tag: int
    values: dict[str, str]
    source: str

    def take(self, name, default=None):
        """读取并移除字段，省略时返回明确的缺省值。"""
        return self.values.pop(name.lower(), default)

    def done(self, ignored=()):
        """只容许已明确分类为元数据的余项，其他字段均拒绝。"""
        for key in ignored:
            self.values.pop(key, None)
        require(not self.values, f"{self.source}: 不支持的{self.kind}字段: {', '.join(self.values)}")


def tokens(value):
    """分词并去除SIF常量类型前缀，拒绝变量、过程和表达式。"""
    require(value is not None, "SIF缺少必需字段")
    result = shlex.split(value, comments=False)
    if result and result[0].lower() in ("real", "integer", "logical", "string"):
        result.pop(0)
    require(result and not any(v.lower() in ("variable", "procedure", "matc") or '$' in v for v in result),
            f"SIF只支持常量字段: {value}")
    return result


def scalar(value):
    """读取恰好一个常数实数。"""
    parts = tokens(value)
    require(len(parts) == 1, f"需要常量标量，拒绝表格/张量: {value}")
    return number(parts[0])


def ids(value):
    """读取常量正整数标签数组。"""
    return [integer(v) for v in tokens(value)]


def parseSif(path):
    """解析逐行封闭SIF节；数组声明检查长度，重复字段和节拒绝。"""
    sections = []
    current = None
    kinds = r"Header|Simulation|Constants|Body Force|Boundary Condition|Initial Condition|Body|Equation|Solver|Material"
    for lineNo, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        # SIF的!注释只在引号外生效。
        line = re.split(r'!(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)', raw, maxsplit=1)[0].strip()
        if not line:
            continue
        where = f"{path}:{lineNo}"
        if current is None and (line.upper() == "RUN" or re.fullmatch(r'Solver\s+\d+\s*::\s*Reference Norm\s*=\s*Real\s+[\d.eEdD+\-]+', line, re.I)):
            continue  # 上游测试驱动元数据，不执行RUN或读取参考范数参与求解。
        if line.lower() == "end":
            require(current is not None, f"{where}: 孤立End")
            sections.append(current)
            current = None
            continue
        match = re.fullmatch(f"({kinds})(?:\\s+(\\d+))?", line, flags=re.I)
        if match:
            require(current is None, f"{where}: 未结束的节")
            key = match[1].lower()
            tag = integer(match[2]) if match[2] else 0
            require(not any(s.kind == key and s.tag == tag for s in sections), f"{where}: 重复节")
            current = Section(key, tag, {}, where)
            continue
        require(current is not None, f"{where}: 不支持的SIF指令（Include/MATC/RUN均不执行）: {line}")
        if '=' in line:
            key, value = line.split('=', 1)
            key = key.strip().lower()
            array = re.fullmatch(r"(.+?)\((\d+)\)", key)
            if array:
                key = array[1].strip()
                require(len(tokens(value)) == int(array[2]), f"{where}: SIF数组长度不匹配")
        else:
            match = re.fullmatch(r'(Mesh DB|Include Path|Results Directory|CHECK KEYWORDS)\s+(.+)', line, re.I)
            require(match is not None, f"{where}: 不支持的SIF语句: {line}")
            key, value = match[1].lower(), match[2]
        require(key not in current.values or key == "output intervals", f"{where}: 重复SIF字段: {key}")
        current.values[key] = value.strip()
    require(current is None, f"{path}: 未结束的SIF节")
    return sections


def readMesh(directory, dimension):
    """读取Elmer串行原生mesh.*，核对表头、ID、单元类型与外边界身份。"""
    paths = {name: directory / f"mesh.{name}" for name in ("header", "nodes", "elements", "boundary")}
    for path in paths.values():
        require(path.is_file(), f"缺少Elmer网格文件: {path}")
    rows = {name: [line.split() for line in path.read_text().splitlines() if line.strip()]
            for name, path in paths.items()}
    header = rows["header"]
    require(len(header) >= 2 and len(header[0]) == 3 and len(header[1]) == 1, "Elmer网格表头格式错误")
    counts = list(map(int, header[0]))
    require(counts == [len(rows[k]) for k in ("nodes", "elements", "boundary")], "Elmer网格记录数与表头不一致")
    types = {int(row[0]): int(row[1]) for row in header[2:] if len(row) == 2}
    require(len(header) == 2 + int(header[1][0]) and len(types) == int(header[1][0]), "Elmer网格类型表不完整")
    model = Model(dimension)
    model.inputs.update(paths.values())
    for row in rows["nodes"]:
        require(len(row) == 5, "Elmer节点记录必须包含ID、分区占位及三坐标")
        tag = integer(row[0])
        require(tag not in model.nodes, f"Elmer重复节点ID: {tag}")
        require(row[1] in ('-1', '0', '1'), "不支持分区节点")
        model.nodes[tag] = tuple(number(v) for v in row[2:])
    bodies, boundaries = defaultdict(set), defaultdict(set)
    actualTypes = defaultdict(int)
    for row in rows["elements"]:
        require(len(row) >= 3, "Elmer单元记录不完整")
        tag, body, kind = (integer(v) for v in row[:3])
        require(kind == (303 if dimension == 2 else 504), f"Elmer单元{tag}类型{kind}不受支持；不降阶或自动剖分")
        require(len(row) == 3 + dimension + 1 and tag not in model.cells, "Elmer单元节点数错误或重复ID")
        model.cells[tag] = tuple(integer(v) for v in row[3:])
        bodies[body].add(tag)
        actualTypes[kind] += 1
    facetMap = model.facets()
    seen = set()
    for row in rows["boundary"]:
        require(len(row) >= 5, "Elmer边界记录不完整")
        tag, boundary = integer(row[0]), integer(row[1])
        left, right, kind = map(int, row[2:5])
        require(tag not in seen, "Elmer边界ID重复")
        seen.add(tag)
        require(kind == (202 if dimension == 2 else 303) and len(row) == 5 + dimension,
                f"不支持的Elmer边界类型: {kind}")
        face = tuple(sorted(integer(v) for v in row[5:]))
        require(face in facetMap, f"Elmer边界不属于单元侧: {face}")
        declared = {v for v in (left, right) if v != 0}
        require(not declared or declared == set(facetMap[face]), "Elmer边界母单元记录不一致")
        boundaries[boundary].add(face)
        actualTypes[kind] += 1
    require(dict(actualTypes) == types, "Elmer网格类型计数不一致")
    return model, bodies, boundaries


def convert(path, options):
    """将封闭SIF常量纯导热问题转换为独立中立模型。

    可显式选择后向Euler重离散，记录该适配；未选择时拒绝BDF2及分段时间步。
    """
    sections = parseSif(path)
    byKind = defaultdict(dict)
    for section in sections:
        byKind[section.kind][section.tag] = section

    def unique(kind):
        require(len(byKind[kind]) == 1, f"SIF必须包含唯一{kind}节")
        return next(iter(byKind[kind].values()))

    header, simulation = unique("header"), unique("simulation")
    meshDb = shlex.split(header.take("mesh db", '"." "mesh"'))
    require(len(meshDb) == 2, "Mesh DB须包含根目录和网格目录")
    directory = options.meshDir or path.parent.joinpath(*meshDb)
    header.done(("check keywords", "include path", "results directory"))
    coordinates = tokens(simulation.take("coordinate system"))
    coordinateSystem = " ".join(coordinates).lower()
    require(coordinateSystem in ("cartesian 2d", "cartesian 3d"), "仅支持Cartesian 2D/3D")
    dimension = 2 if coordinateSystem.endswith("2d") else 3
    mapping = simulation.take("coordinate mapping")
    require(mapping is None or ids(mapping) == [1, 2, 3], "不支持坐标轴重排")
    model, meshBodies, meshBoundaries = readMesh(directory, dimension)
    model.inputs.add(path)
    model.thickness = options.thickness if dimension == 2 else None
    if dimension == 2:
        require(options.thickness is not None, "Elmer二维模型请通过--thickness明确厚度[m]，每单位厚度问题可填1")
    analysis = " ".join(tokens(simulation.take("simulation type"))).lower()
    require(analysis in ("steady state", "transient"), f"不支持的Simulation Type: {analysis}")
    if analysis == "transient":
        method = " ".join(tokens(simulation.take("timestepping method", "BDF"))).lower()
        order = simulation.take("bdf order", "1")
        steps = [number(v) for v in tokens(simulation.take("timestep sizes"))]
        intervals = ids(simulation.take("timestep intervals"))
        require(len(steps) == len(intervals) and all(v > 0 for v in steps), "时间步数组不匹配或非正")
        start = scalar(simulation.take("time", "0"))
        duration = sum(dt*n for dt, n in zip(steps, intervals))
        if options.backwardEulerStep is None:
            require(method == "bdf" and scalar(order) == 1 and len(set(steps)) == 1,
                    "时间格式不能等价转换：仅原生BDF1等步长；可显式使用--backward-euler-step重离散")
            dt = steps[0]
        else:
            dt = options.backwardEulerStep
            model.notes.append(f"调用者明确重离散：源{method} order={order}, steps={steps}, intervals={intervals}；目标后向Euler dt={dt}s。")
        require(dt > 0 and math.isfinite(dt) and math.isfinite(duration) and duration > 0, "时间范围非法")
        model.time = {"start": start, "end": start+duration, "step": dt, "method": "backwardEuler",
                      "maxSteps": stepLimit(duration, dt)}
    else:
        require(options.backwardEulerStep is None and options.initialTemperature is None,
                "稳态不接受瞬态重离散或初温默认值选项")
    simulation.done(("max output level", "steady state max iterations", "output intervals", "post file", "output file"))
    for constants in byKind["constants"].values():
        constants.done(("gravity", "stefan boltzmann"))
    used = defaultdict(set)
    allBodies = set()
    initialByNode = {}
    for bodyId, body in byKind["body"].items():
        target = body.take("target bodies")
        targets = ids(target) if target else [bodyId]
        require(set(targets) <= meshBodies.keys(), f"Body引用未知网格体: {targets}")
        require(not allBodies.intersection(targets), "SIF多个Body覆盖相同网格体")
        allBodies.update(targets)
        cells = set().union(*(meshBodies[t] for t in targets))
        materialId = ids(body.take("material"))
        equationId = ids(body.take("equation"))
        require(len(materialId) == len(equationId) == 1, "Body须引用单一材料和方程")
        mId, eId = materialId[0], equationId[0]
        require(mId in byKind["material"] and eId in byKind["equation"], "Body引用缺失材料/方程")
        used["material"].add(mId)
        used["equation"].add(eId)
        name = f"material-{mId}"
        model.assignments.setdefault(name, set()).update(cells)
        sourceRef = body.take("body force")
        if sourceRef:
            refs = ids(sourceRef)
            require(len(refs) == 1 and refs[0] in byKind["body force"], "缺失Body Force")
            used["body force"].add(refs[0])
            force = byKind["body force"][refs[0]]
            forceValues = dict(force.values)
            massSource = scalar(force.take("heat source", "0"))
            volumeSource = scalar(force.take("volumetric heat source", "0"))
            require(not (massSource and volumeSource), "Elmer体积源优先语义：不同时接受两种非零热源")
            densityRaw = byKind["material"][mId].values.get("density")
            require(not massSource or densityRaw is not None, "Heat Source按W/kg解释，必须有显式密度")
            q = volumeSource if volumeSource else massSource * (scalar(densityRaw) if densityRaw else 1)
            force.done(("name",))
            force.values = forceValues
            if q:
                model.sources.append((cells, q))
        icRef = body.take("initial condition")
        initialValue = options.initialTemperature
        if icRef:
            refs = ids(icRef)
            require(len(refs) == 1 and refs[0] in byKind["initial condition"], "缺失Initial Condition")
            used["initial condition"].add(refs[0])
            ic = byKind["initial condition"][refs[0]]
            variable = "temperature"
            # 变量名在下面Solver验证前从活动HeatSolver读取。
            activeEquation = byKind["equation"][eId]
            active = ids(activeEquation.values.get("active solvers"))
            require(len(active) == 1 and active[0] in byKind["solver"], "仅接受单一活动HeatSolver")
            variable = " ".join(tokens(byKind["solver"][active[0]].values.get("variable", 'Temperature'))).lower()
            require(variable in ic.values, f"Initial Condition缺少变量{variable}")
            initialValue = scalar(ic.values[variable])
            require(set(ic.values) <= {variable, "name"}, "Initial Condition包含不支持的物理字段")
        if model.time:
            require(initialValue is not None, "瞬态缺少初温；请提供Initial Condition或--initial-temperature")
            value = kelvin(initialValue, options.temperatureUnit)
            for cell in cells:
                model.initial[cell] = value
                for node in model.cells[cell]:
                    require(node not in initialByNode or initialByNode[node] == value, "相邻体初温在共享节点冲突")
                    initialByNode[node] = value
        body.done(("name",))
    require(allBodies == meshBodies.keys(), "SIF Body未覆盖所有网格体")
    variables = set()
    for eId in used["equation"]:
        equation = byKind["equation"][eId]
        active = ids(equation.take("active solvers"))
        require(len(active) == 1 and active[0] in byKind["solver"], "仅支持单一活动HeatSolver")
        used["solver"].add(active[0])
        equation.done(("name",))
    for sId, solver in byKind["solver"].items():
        if sId not in used["solver"]:
            require(" ".join(tokens(solver.values.get("exec solver", 'always'))).lower() == "never", "未引用且未禁用的Solver被拒绝")
            model.notes.append(f"不复制已禁用的Solver {sId}。")
            continue
        procedure = tokens(solver.take("procedure"))
        require([p.lower() for p in procedure] == ["heatsolve", "heatsolver"], "仅支持标准HeatSolve HeatSolver")
        variable = " ".join(tokens(solver.take("variable", "Temperature"))).lower()
        variables.add(variable)
        for key in list(solver.values):
            if key.startswith(("linear system ", "nonlinear system ")) or key in (
                    "equation", "stabilize", "optimize bandwidth", "steady state convergence tolerance", "name"):
                solver.take(key)
        solver.done()
    require(len(variables) == 1, "多个温度变量不能合并")
    variable = variables.pop()
    for mId, material in byKind["material"].items():
        require(mId in used["material"], f"未使用材料{mId}，请清理歧义配置")
        result = {"conductivity": scalar(material.take("heat conductivity"))}
        for sourceKey, targetKey in (("density", "density"), ("heat capacity", "specificHeat")):
            raw = material.take(sourceKey)
            if raw is not None:
                result[targetKey] = scalar(raw)
        require(("density" in result) == ("specificHeat" in result), "密度与比热必须同时提供")
        model.materials[f"material-{mId}"] = result
        material.done(("name",))
    for bId, boundary in byKind["boundary condition"].items():
        targets = ids(boundary.take("target boundaries"))
        require(set(targets) <= meshBoundaries.keys(), "Boundary Condition引用未知网格边界")
        faces = set().union(*(meshBoundaries[t] for t in targets))
        rawTemperature = boundary.take(variable)
        if rawTemperature is not None:
            model.boundaries.append((faces, {"kind": "temperature", "temperature": constant(kelvin(scalar(rawTemperature), options.temperatureUnit))}))
        rawFlux = boundary.take("heat flux")
        fluxFlag = boundary.take("heat flux bc")
        require(fluxFlag is None or " ".join(tokens(fluxFlag)).lower() == "true", "只接受启用的Heat Flux BC")
        if rawFlux is not None:
            model.boundaries.append((faces, {"kind": "outwardFlux", "flux": constant(-scalar(rawFlux))}))
        h, ambient = boundary.take("heat transfer coefficient"), boundary.take("external temperature")
        require((h is None) == (ambient is None), "对流必须同时提供系数和环境温度")
        require((rawFlux is None and h is None) or fluxFlag is not None,
                "Elmer热流/对流必须显式Heat Flux BC=True，不能把未启用的载荷自动激活")
        if h is not None:
            coefficient = scalar(h)
            require(coefficient >= 0, "对流系数必须非负")
            model.boundaries.append((faces, {"kind": "convection", "coefficient": constant(coefficient),
                                           "ambientTemperature": constant(kelvin(scalar(ambient), options.temperatureUnit))}))
        boundary.done(("name",))
    for kind in ("body force", "initial condition", "equation"):
        require(set(byKind[kind]) == used[kind], f"存在未引用的{kind}节")
    model.notes += ["Elmer Heat Source按质量热源W/kg乘以材料密度，Volumetric Heat Source按W/m^3。",
                    "Elmer Heat Flux为输入热流，转换为EmberField外向热流时反号。",
                    "上游测试驱动RUN和Solver :: Reference Norm仅为测试元数据，不执行、不转换为求解设置。",
                    f"调用者声明源温度单位{options.temperatureUnit}，其他物理量及网格坐标均为SI。"]
    return model
