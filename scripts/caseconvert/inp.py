"""CalculiX及兼容Abaqus关键字子集的常量热传导转换，不执行用户子程序。"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import csv
import math

from .common import Model, constant, integer, kelvin, number, require, stepLimit


@dataclass
class Card:
    """一个关键字、封闭参数字典及来源标记的数据行。"""
    keyword: str
    parameters: dict[str, str | None]
    rows: list[list[str]]
    source: str

    def check(self, allowed):
        """拒绝未知参数，不忽略会改变物理含义的关键字修饰。"""
        require(set(self.parameters) <= set(allowed), f"{self.source}: {self.keyword}不支持的参数: {set(self.parameters)-set(allowed)}")


def fields(line):
    """按CSV方式处理逗号与双引号；保留中间空字段。"""
    result = next(csv.reader([line], skipinitialspace=True))
    return [v.strip() for v in result]


def readDeck(path, inputs, stack=()):
    """展开相对INCLUDE并保留来源；循环、超过64层和重复参数拒绝。"""
    path = path.resolve()
    require(path not in stack and len(stack) < 64, f"INCLUDE循环或过深: {path}")
    require(path.is_file(), f"INCLUDE或输入文件不存在: {path}")
    inputs.add(path)
    cards = []
    current = None
    for lineNo, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith('**'):
            continue
        where = f"{path}:{lineNo}"
        if line.startswith('*'):
            items = fields(line[1:])
            require(items[0] and items[-1], f"{where}: 不支持跨行关键字或空参数")
            parameters = {}
            for item in items[1:]:
                pair = item.split('=', 1)
                key = pair[0].strip().upper()
                require(key and key not in parameters, f"{where}: 重复或空关键字参数")
                parameters[key] = pair[1].strip().strip('"') if len(pair) == 2 else None
            current = Card(items[0].upper(), parameters, [], where)
            if current.keyword == 'INCLUDE':
                current.check({'INPUT'})
                require(parameters.get('INPUT'), f"{where}: INCLUDE必须指定INPUT")
                cards.extend(readDeck(path.parent / parameters['INPUT'], inputs, (*stack, path)))
                current = None
            else:
                cards.append(current)
        else:
            require(current is not None, f"{where}: 游离数据行或跨文件关键字数据不受支持")
            current.rows.append(fields(line))
    return cards


def convert(path, options):
    """转换单步骤纯导热INP，严格核对材料、集合、侧编号及节点温度可表示性。"""
    inputs = set()
    cards = readDeck(path, inputs)
    nodes, cells = {}, {}
    nodeSets, elementSets = defaultdict(list), defaultdict(list)
    materials = {}
    sections, loads, constraints, initialCards = [], [], [], []
    stepCount, inStep, stepEnded = 0, False, False
    procedure = None
    material = None
    dimensions = set()
    notes = []

    def parameter(card, key):
        require(card.parameters.get(key), f"{card.source}: 缺少参数{key}")
        return card.parameters[key].upper()

    for card in cards:
        key, p, rows = card.keyword, card.parameters, card.rows
        require(not stepEnded, f"{card.source}: END STEP之后不支持其他关键字")
        if key == 'STEP':
            card.check({'NAME', 'INC'})
            require(not inStep and stepCount == 0 and not rows, f"{card.source}: 只支持一个分析步")
            stepCount += 1
            inStep = True
            material = None
        elif key == 'END STEP':
            card.check(set())
            require(inStep and not rows, f"{card.source}: END STEP位置非法")
            inStep, stepEnded = False, True
        elif key == 'HEAT TRANSFER':
            require(inStep and procedure is None, f"{card.source}: 需要唯一HEAT TRANSFER步骤")
            card.check({'STEADY STATE', 'DIRECT', 'SOLVER'})
            procedure = card
        elif key == 'NODE':
            require(not inStep, f"{card.source}: NODE只能位于模型定义区")
            card.check({'NSET'})
            for row in rows:
                require(2 <= len(row) <= 4, f"{card.source}: NODE须含ID和1至3坐标")
                tag = integer(row[0])
                require(tag not in nodes, f"{card.source}: 重复节点{tag}")
                nodes[tag] = tuple([number(v) for v in row[1:]] + [0.] * (4-len(row)))
                if 'NSET' in p:
                    nodeSets[parameter(card, 'NSET')].append(str(tag))
        elif key == 'ELEMENT':
            require(not inStep, f"{card.source}: ELEMENT只能位于模型定义区")
            card.check({'TYPE', 'ELSET'})
            kind = parameter(card, 'TYPE')
            supported = {'C3D4': 3, 'DC3D4': 3, 'CPS3': 2, 'CPE3': 2, 'DC2D3': 2}
            require(kind in supported, f"{card.source}: 不支持单元{kind}；不降阶或自动剖分")
            dim = supported[kind]
            dimensions.add(dim)
            for row in rows:
                require(len(row) == dim + 2, f"{card.source}: {kind}节点数错误，跨行连通不受支持")
                tag = integer(row[0])
                require(tag not in cells, f"重复单元{tag}")
                cells[tag] = tuple(integer(v) for v in row[1:])
                if 'ELSET' in p:
                    elementSets[parameter(card, 'ELSET')].append(str(tag))
        elif key in ('NSET', 'ELSET'):
            require(not inStep, f"{card.source}: 集合必须位于模型定义区")
            card.check({key, 'GENERATE'})
            target = nodeSets if key == 'NSET' else elementSets
            name = parameter(card, key)
            for row in rows:
                values = [v.upper() for v in row if v]
                if 'GENERATE' in p:
                    require(len(values) in (2, 3), f"{card.source}: GENERATE需要起点、终点、可选步长")
                    start, end = integer(values[0]), integer(values[1])
                    stride = integer(values[2]) if len(values) == 3 else 1
                    require(start <= end and (end-start) % stride == 0, "GENERATE范围非法")
                    require((end-start)//stride < 100000000, "GENERATE展开记录过大")
                    target[name].extend(str(v) for v in range(start, end+1, stride))
                else:
                    target[name].extend(values)
        elif key == 'MATERIAL':
            require(not inStep and not rows, f"{card.source}: MATERIAL位置或数据非法")
            card.check({'NAME'})
            material = parameter(card, 'NAME')
            require(material not in materials, "重复材料名")
            materials[material] = {}
        elif key in ('CONDUCTIVITY', 'DENSITY', 'SPECIFIC HEAT'):
            require(not inStep and material is not None, f"{card.source}: 热物性缺少MATERIAL上下文")
            card.check({'TYPE'} if key == 'CONDUCTIVITY' else set())
            require('TYPE' not in p or (p['TYPE'] or '').upper() == 'ISO', "不支持各向异性导热率")
            require(len(rows) == 1 and len(rows[0]) == 1, f"{card.source}: 只支持单个与温度无关的常数物性")
            target = {'CONDUCTIVITY': 'conductivity', 'DENSITY': 'density', 'SPECIFIC HEAT': 'specificHeat'}[key]
            require(target not in materials[material], f"重复热物性: {target}")
            materials[material][target] = number(rows[0][0])
        elif key == 'SOLID SECTION':
            require(not inStep, f"{card.source}: SOLID SECTION位置非法")
            card.check({'ELSET', 'MATERIAL'})
            sections.append((parameter(card, 'ELSET'), parameter(card, 'MATERIAL'), rows))
            material = None
        elif key == 'BOUNDARY':
            card.check(set())
            constraints.extend((card.source, row) for row in rows)
        elif key == 'INITIAL CONDITIONS':
            require(not inStep, f"{card.source}: 初始条件必须位于STEP之前")
            card.check({'TYPE'})
            require(parameter(card, 'TYPE') == 'TEMPERATURE', "只支持温度初态")
            initialCards.extend((card.source, row) for row in rows)
        elif key in ('DFLUX', 'FILM'):
            require(inStep, f"{card.source}: 热载荷须位于STEP中")
            card.check(set())
            loads.append(card)
        elif key in ('HEADING', 'PREPRINT'):
            card.check(set() if key == 'HEADING' else {'ECHO', 'MODEL', 'HISTORY', 'CONTACT'})
            notes.append(f"不复制源显示设置: {key}")
        elif key in ('NODE FILE', 'EL FILE', 'NODE PRINT', 'EL PRINT'):
            card.check({'NSET', 'ELSET', 'FREQUENCY', 'TOTALS', 'GLOBAL', 'TIME POINTS'})
            require(inStep, f"{card.source}: 输出指令须位于STEP")
            notes.append(f"不复制源输出指令: {key}")
        else:
            require(False, f"{card.source}: 不支持的关键字*{key}，不能忽略潜在物理设置")
    require(stepCount == 1 and stepEnded and procedure is not None, "必须有一个完整HEAT TRANSFER分析步")
    require(len(dimensions) == 1, "不支持混合二维与三维单元")
    model = Model(dimensions.pop(), nodes, cells)
    model.inputs = inputs
    model.materials = materials
    model.notes = notes

    def resolve(token, sets, universe, stack=()):
        token = token.upper()
        if token.isdigit():
            tag = integer(token)
            require(tag in universe, f"集合引用不存在ID: {tag}")
            return {tag}
        require(token in sets and token not in stack, f"集合不存在或循环: {token}")
        require(len(stack) < 64, "集合嵌套过深")
        result = set()
        for item in sets[token]:
            result.update(resolve(item, sets, universe, (*stack, token)))
        require(result, f"空集合: {token}")
        return result

    for name in nodeSets:
        resolve(name, nodeSets, nodes)
    for name in elementSets:
        resolve(name, elementSets, cells)
    for region, name, rows in sections:
        require(name in materials, f"截面引用缺失材料: {name}")
        model.assignments.setdefault(name, set())
        members = resolve(region, elementSets, cells)
        require(not model.assignments[name].intersection(members), "重复SOLID SECTION分配")
        model.assignments[name].update(members)
        flat = [v for row in rows for v in row if v]
        if model.dimension == 2:
            require(len(flat) == 1, "二维SOLID SECTION需要唯一明确厚度[m]")
            thickness = number(flat[0])
            require(model.thickness is None or model.thickness == thickness, "不同区域厚度不能由当前统一厚度表达")
            model.thickness = thickness
        else:
            require(not flat, "三维SOLID SECTION不接受额外数据")
    require(set(materials) == set(model.assignments), "存在未使用材料")
    for name, materialValues in materials.items():
        require(('density' in materialValues) == ('specificHeat' in materialValues), f"材料{name}密度与比热须同时提供")
    facets = model.facets()
    exterior = {face for face, owners in facets.items() if len(owners) == 1}
    prescribed = {}
    for where, row in constraints:
        require(len(row) == 4 and (row[1], row[2]) in (('11', '11'), ('0', '0')),
                f"{where}: 仅支持热自由度11（或CalculiX的0）常温BOUNDARY")
        value = kelvin(number(row[3]), options.temperatureUnit)
        for node in resolve(row[0], nodeSets, nodes):
            require(node not in prescribed or prescribed[node] == value, f"节点{node}温度冲突")
            prescribed[node] = value
    byValue = defaultdict(set)
    represented = set()
    for face in exterior:
        if all(n in prescribed for n in face) and len({prescribed[n] for n in face}) == 1:
            value = prescribed[face[0]]
            byValue[value].add(face)
            represented.update(face)
    require(represented == prescribed.keys(), "存在无法由完整外边界侧表达的孤立节点温度约束，拒绝扩张节点集合")
    for value, faces in byValue.items():
        model.boundaries.append((faces, {'kind': 'temperature', 'temperature': constant(value)}))

    def loadFaces(token, label):
        require(len(label) == 2 and label[1].isdigit(), f"不支持的侧标签: {label}")
        faceIndex = int(label[1])-1
        local = [(0, 1), (1, 2), (2, 0)] if model.dimension == 2 else [(0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0)]
        require(0 <= faceIndex < len(local), f"侧编号越界: {label}")
        # 此时还未规范化单元方向，侧号严格按源连通顺序解释。
        return {tuple(sorted(cells[e][j] for j in local[faceIndex])) for e in resolve(token, elementSets, cells)}

    for card in loads:
        for row in card.rows:
            if card.keyword == 'DFLUX':
                require(len(row) == 3, f"{card.source}: DFLUX需要区域、标签、值")
                label = row[1].upper()
                if label == 'BF':
                    model.sources.append((resolve(row[0], elementSets, cells), number(row[2])))
                else:
                    require(label.startswith('S'), "只支持BF和S1..S4的均匀DFLUX")
                    model.boundaries.append((loadFaces(row[0], label), {'kind': 'outwardFlux', 'flux': constant(-number(row[2]))}))
            else:
                require(len(row) == 4 and row[1].upper().startswith('F'), f"{card.source}: FILM需要区域、Fx、环境温度、系数")
                h = number(row[3])
                require(h >= 0, "对流系数必须非负")
                model.boundaries.append((loadFaces(row[0], row[1].upper()), {'kind': 'convection', 'coefficient': constant(h),
                    'ambientTemperature': constant(kelvin(number(row[2]), options.temperatureUnit))}))
    if 'STEADY STATE' not in procedure.parameters:
        require(options.backwardEulerStep is not None, "INP瞬态须显式--backward-euler-step确认在EmberField重离散")
        require(len(procedure.rows) == 1, "瞬态HEAT TRANSFER必须提供单行时间控制")
        row = procedure.rows[0]
        require(2 <= len(row) <= 4 and not any(v for v in row[2:]), "时间控制额外非空字段不能保留")
        duration = number(row[1])
        dt = options.backwardEulerStep
        require(duration > 0 and dt > 0 and math.isfinite(dt), "时间范围和时间步必须正")
        model.time = {'start': 0, 'end': duration, 'step': dt, 'method': 'backwardEuler', 'maxSteps': stepLimit(duration, dt)}
        initial = {}
        for where, row in initialCards:
            require(len(row) == 2, f"{where}: 仅支持节点常数初始温度")
            for node in resolve(row[0], nodeSets, nodes):
                value = kelvin(number(row[1]), options.temperatureUnit)
                require(node not in initial or initial[node] == value, "初始节点温度冲突")
                initial[node] = value
        if options.initialTemperature is not None:
            value = kelvin(options.initialTemperature, options.temperatureUnit)
            for node in nodes:
                initial.setdefault(node, value)
        for cell, connected in cells.items():
            require(all(n in initial for n in connected), "缺少初始节点温度，可显式--initial-temperature提供默认值")
            values = {initial[n] for n in connected}
            require(len(values) == 1, "空间变化的单元内初始温度不能由当前常量区域初态等价表示")
            model.initial[cell] = values.pop()
        model.notes.append(f"调用者明确将INP瞬态时间积分重离散为后向Euler dt={dt}s，源时间控制={procedure.rows}。")
    else:
        require(options.backwardEulerStep is None, "稳态不接受瞬态重离散选项")
        require(not initialCards and options.initialTemperature is None, "稳态不转换初始温度场")
        require(len(procedure.rows) <= 1, "稳态时间控制行过多")
        model.notes.append("稳态仅保留最终静态载荷，不复制源步骤伪时间的载荷渐增过程。")
    model.notes += [f"调用者声明源温度单位{options.temperatureUnit}，其他物理量及坐标均为SI。",
                    "INP均匀DFLUX的BF按W/m^3，Sx为输入热流W/m^2，转换外向热流时反号。",
                    "节点温度约束仅在能由完整外侧精确重建相同节点集合时转换。"]
    return model
