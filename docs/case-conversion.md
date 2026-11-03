# 外部热传导算例转换

## 格式选择

热仿真输入不只是网格。材料、物理区域、边界、初始状态和时间控制共同定义算例，
只将网格转换成 MSH 不能完成算例迁移。

| 程序或格式 | 输入组织与特点 | 本工具的处理 |
| --- | --- | --- |
| Elmer | SIF 描述方程、材料、体、边界和求解控制；原生 `mesh.*` 保存网格及标签 | 提供 `elmer` 适配器，适合迁移常量纯导热模型 |
| CalculiX，兼容部分 Abaqus INP 关键字 | INP 用关键字描述节点、单元、集合、材料、载荷及分析步 | 提供 `inp` 适配器，按本文封闭子集解析；不是通用 Abaqus 导入器 |
| OpenFOAM | 多面体有限体积网格、patch 边界以及分散的场和热物性字典 | 本次不提供自动转换；多面体到 P1 单纯形网格、单元中心到节点场和材料模型均需额外映射，不能只改字段名 |

选择前两种格式，是因为它们能明确提供有限元连通关系、区域和纯导热载荷，
且官方格式说明与开源算例可用于核对。商业软件的完整模型数据库不在此范围。

## 使用方法

需要 Python **3.10 或更新版本**，只使用标准库，无需 pip 安装依赖、Elmer 或 CalculiX。
转换是 CPU 离线预处理，生成后仍由 EmberField 的 GPU 路径组装和求解。

```bash
python3 scripts/convert_case.py inp example/conversion/tetra-source.inp \
  --temperature-unit K --output data/imported-tetra
build-output/emberfield data/imported-tetra/case.json \
  --output-root data --output imported-tetra-result

python3 scripts/convert_case.py elmer example/conversion/elmer/heat.sif \
  --temperature-unit K --thickness 1 --output data/imported-elmer
```

安装后入口为 `<prefix>/share/EmberField/scripts/convert_case.py`，旁边的
`caseconvert/` 是随工具安装的适配模块，不要单独移动入口文件。

| 参数 | 含义 |
| --- | --- |
| `elmer` / `inp` | 必选源格式，随后给出主输入文件路径 |
| `--output DIR` | 必选，新建转换算例目录；已存在则拒绝，不覆盖 |
| `--temperature-unit K\|C` | 必选，声明源温度单位；摄氏温度加 273.15 转换为 K |
| `--mesh-dir DIR` | Elmer 原生网格目录；省略时相对于 SIF 解析 `Mesh DB` |
| `--thickness VALUE` | Elmer 二维统一厚度，单位 m，必须显式给出；三维不需要。INP 厚度来自 `SOLID SECTION`，不接受本参数 |
| `--initial-temperature VALUE` | 显式补充未给出的瞬态初温，单位与源温度声明一致；不覆盖源初温 |
| `--backward-euler-step DT` | 显式同意瞬态重离散为后向 Euler，并设置步长 s；不表示复现原积分方法 |

**除温度外，坐标及所有物理量必须事先采用 SI**：m、s、kg、W、J。
INP 本身没有固定单位制，脚本不能从数值猜测 mm 或 m，也不会自动换算导热率、密度、比热。
温差和热流不随摄氏/K 的零点平移而改变。

## 支持的物理子集

共同支持二维 XY 平面的 Triangle3、三维 Tet4，连续 P1 标量导热，
区域常数的正各向同性导热率、密度和比热，稳态或瞬态，体积热源、规定温度、
给定热流和线性对流。不自动降阶或剖分其他单元。
二维必须统一正厚度，密度与比热需成对提供，瞬态必须具备热容量。

| 来源 | 源字段或结构 | EmberField 对应与检查 |
| --- | --- | --- |
| Elmer | `Body` / `Target Bodies` / `Material` / `Equation` | 每个活动体恰有一个材料与标准 HeatSolver；材料恰好覆盖全部域单元 |
| Elmer | `Heat Conductivity`、`Density`、`Heat Capacity` | `conductivity`、`density`、`specificHeat` 常量字段 |
| Elmer | `Heat Source` / `Volumetric Heat Source` | 前者 W/kg 乘以显式密度，后者 W/m³；拒绝同时非零 |
| Elmer | `Temperature` 或活动 HeatSolver 的变量名 | 完整外侧上的规定温度 |
| Elmer | `Heat Flux` | 输入热流反号为 `outwardFlux`，必须显式 `Heat Flux BC=True` |
| Elmer | `Heat Transfer Coefficient` + `External Temperature` | 线性对流，二者须配对且 `Heat Flux BC=True` |
| Elmer | 原生 `mesh.header/nodes/elements/boundary` | 校验记录数、类型数、ID、连接及侧身份；允许边界母单元为 0 后从拓扑推导 |
| INP | `NODE`、`ELEMENT`、`NSET`、`ELSET` | 支持集合引用、GENERATE、相对 INCLUDE；循环与超过 64 层嵌套报错 |
| INP | `C3D4/DC3D4`、`CPS3/CPE3/DC2D3` | 线性四面体或三角形；这些机械单元名称仅在独立 `HEAT TRANSFER` 步骤中解释为导热拓扑 |
| INP | `MATERIAL`、`CONDUCTIVITY`、`DENSITY`、`SPECIFIC HEAT`、`SOLID SECTION` | 常量热物性及材料分配；二维截面显式厚度必须一致 |
| INP | `BOUNDARY` 的自由度 11 或 CalculiX 的 0 | 仅当完整外侧能精确重建相同受约束节点集合时转换；孤立节点或内部节点约束报错 |
| INP | `DFLUX` 的 BF / S1..S4 | BF 为体积热源，Sx 输入热流反号；二维使用 S1..S3 |
| INP | `FILM` 的 F1..F4 | 环境温度与非负换热系数；二维使用 F1..F3 |

源四面体侧号按连接顺序解释：1=(1,2,3)，2=(1,4,2)，3=(2,4,3)，4=(3,4,1)；
三角形侧号为 (1,2)、(2,3)、(3,1)。工具先确定源侧身份，再规范化负向单元连接顺序，
不会因方向调整把载荷移到另一侧。

瞬态原生支持 Elmer 等步长 BDF1。BDF2、分段步长以及 INP 时间积分必须用
`--backward-euler-step` 明确选择目标方法；保留起止物理时间，末步由 EmberField 截短。
初态必须覆盖所有单元，同一单元及共享节点的温度必须一致，且与规定温度相容。
这意味着空间线性变化初态和初始边界跳变目前不能等价转换。
稳态 INP 仅保留最终静态载荷，不复制伪时间渐增路径。

## 输出与失败语义

一个成功目录包含：

| 文件 | 字段或结构 |
| --- | --- |
| `case.json` | schemaVersion=1；mesh、analysis、materials、materialAssignments、sources、boundaries、boundaryPolicy、discretization、linearSolver、output；瞬态增加 time、initialState。字段完整定义见[算例配置](case.md) |
| `mesh.msh` | MSH 4.1 ASCII；PhysicalNames、Entities、Nodes、Elements。保留域节点/单元 ID，生成受物理条件引用的边界单元；按完整物理组成员签名分实体，实现交叠分组而不复制体单元 |
| `conversion-report.json` | schemaVersion=1，sourceFormat、sourceName、units、dimension、nodeCount、cellCount、notes、inputs、case、mesh；inputs 各项为原始路径 path 和完整文件 SHA256；notes 记录单位声明、转换规则、显式重离散及未复制的求解/输出元数据 |

网格输出是求解模型，不是源文件的无损归档；未使用的源集合、CAD 分类、未施加载荷的
边界标签和源程序输出命令不复制。需要溯源时保留原输入文件和转换报告。
目标求解配置统一采用 CG＋恒等预条件、相对容差 1e-10、绝对容差 1e-12、最多 10000 次迭代，
不复制源程序的线性求解器；用户可在生成后按问题条件修改。
未设置边界默认为绝热，自然条件交叠按加和解释；纯 Neumann 稳态问题仍可能奇异，
转换成功不代表线性系统一定可解或一定收敛。

未知关键字、未知物理参数、辐射、温变物性、各向异性、相变、流热耦合、用户子程序、
动态载荷、多个分析步、高阶或六面体单元均明确失败，不静默删除。
Elmer MATC、过程、Include 和 RUN 不执行；只识别上游测试驱动的独立 RUN 与 Reference Norm 元数据。
INP INCLUDE 只读取数据文件。
源程序纯求解控制和输出元数据按显式白名单忽略，并在报告说明目标设置。

成功退出码为 0；输入错误、能力不匹配或文件错误为 2，错误信息写标准错误。
发布前先完成模型校验；在输出父目录建立临时目录，写完三个文件再重命名，
失败清理自身临时目录，不覆盖已有算例。输入文件须在整个转换期间保持不变。

## 验证方法

`test/IO/Case/test_conversion.py` 使用标准库 unittest，覆盖解析、集合、单位、热源、热流、
对流、瞬态显式适配、拓扑与拒绝语义；传入 `--executable build-output/emberfield` 后，
将转换产物交给真实 CLI，通过四面体体源、表面热流、对流、均匀瞬态温升以及 Elmer 三角形
的解析答案检查整个转换—组装—求解链。

真实 Elmer `heateq_bdf2/TempDist.sif` 还用于验证带显式网格与时间适配的导入流程。
它使用经 ElmerGrid 适配的三角形原生网格，并明确选择后向 Euler；这不等于复现源程序的
四边形/BDF2 数值结果。官方 CalculiX 高阶与辐射样例用于验证拒绝路径。

## 格式依据

- Elmer [Models Manual](https://www.nic.funet.fi/pub/sci/physics/elmer/doc/ElmerModelsManual.pdf)：Heat Equation 的热源与边界字段。
- Elmer [Solver Manual](https://www.nic.funet.fi/pub/sci/physics/elmer/doc/ElmerSolverManual.pdf)：原生网格文件格式附录。
- CalculiX [官方手册源码](https://github.com/Dhondtguido/CalculiX/blob/master/doc/CalculiX.tex)：INCLUDE、集合、热材料、HEAT TRANSFER、BOUNDARY、DFLUX、FILM 与 SOLID SECTION。
- OpenFOAM v13 [Mesh description](https://doc.cfd.direct/openfoam/user-guide-v13/mesh-description)、[Thermophysical models](https://doc.cfd.direct/openfoam/user-guide-v13/thermophysical)：多面体与热物性模型组织。

第三方原文版权属于各自权利人，未将其手册及上游算例加入本项目发行包。
