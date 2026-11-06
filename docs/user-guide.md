# EmberField 用户手册

本手册面向运行算例和在 CUDA/C++20 程序中调用 EmberField 的用户。当前可执行热模型为二维 Triangle3 或三维 Tet4 连续一阶温度场，支持稳态与固定步长后向 Euler 瞬态。MSH 读写器可表达更多网格类型，但读入成功不表示这些类型都能参与热求解。

## 快速开始

按根目录 [README](../README.md) 完成构建；下面的命令从项目根目录执行。示例网格与配置文件位于 [example/cases](../example/cases/)。

~~~bash
./build/emberfield example/cases/steady.json --output-root ./results --output plate-steady
./build/emberfield example/cases/transient.json --output-root ./results --output plate-transient
~~~

运行成功后查看结果目录中的 run-summary.json 和 steps.csv；温度场位于 fields/*.vtu。瞬态的 temperature.pvd 可直接在 ParaView 中打开并按物理时间浏览。已有非空结果目录默认不会被覆盖。

## 命令行

~~~text
emberfield <case.json> [--output-root PATH] [--output NAME] [--resource-limits PATH] [--ui auto|tui|plain] [--no-wait]
emberfield --help
emberfield --version
~~~

| 参数 | 必需 | 含义与默认值 |
| --- | --- | --- |
| case.json | 是 | UTF-8 JSON 算例文件；其内部相对网格路径相对于算例文件所在目录解析。 |
| --output-root PATH | 否 | 结果根目录，默认是实际可执行文件所在目录；显式相对路径相对于启动工作目录解析。 |
| --output NAME | 否 | 根目录下的单级文件夹名，默认是算例文件去掉最后扩展名后的名字。例如 plate.v2.json 对应 plate.v2。不得包含路径分隔符或系统保留名称。 |
| --resource-limits PATH | 否 | 单独的资源预算 JSON 文件；相对路径相对于启动工作目录解析。程序不自动加载项目的 limits.json。 |
| --ui auto/tui/plain | 否 | 默认auto；输入、输出均为TTY且TERM非dumb，窗口至少40列18行时使用TUI，否则普通输出。显式tui不可用时计算前报错。 |
| --no-wait | 否 | TUI完成后自动退出；默认完成后留屏，Enter/Q/Esc关闭。普通输出不等待。 |
| --help / --version | 否 | 显示帮助或程序版本；不要求初始化 CUDA。 |

最终目录始终由 --output-root 与 --output 组合，覆盖算例内的 output.directory；其他输出选项仍由算例决定。如果想复用同一算例，给每次运行不同的 --output 名称，或者在算例中显式设置 output.overwrite。资源文件只改变预算，不改变材料、时间步或求解容差。

退出码：0 成功；2 输入、绑定或离散准备失败；3 线性求解或接受残差失败；1 资源、运行或输出错误；130 在安全步边界响应中断。参数语法错误沿用 CLI11 的退出码。程序同时在结果目录保留结构化运行记录；不要解析终端进度文本作为机器协议。

## 终端进度与取消

TUI沿用demo的六行暖色字符画与副标题。至少85列、24行时显示完整字符画及左右双面板；旋转指示器与当前阶段位于右侧CASE面板的Status行，完成后改为终态标记。较窄或不足24行的窗口使用紧凑标题；32行起显示详细进度，36行起增加最近消息。
窗口缩到40列18行以下时恢复终端并继续普通输出，不重新计算。`-DEMBERFIELD_BUILD_TUI=OFF`
可构建普通CLI；FTXUI仅为CLI私有依赖，不影响下游库用户。

| 阶段 | 展示方式 |
| --- | --- |
| 读取、准备、符号图、建系统、预条件准备、验收、后处理、写出、终结 | 旋转指示器，不估计剩余时间 |
| 数值组装体/自然项及实际重算的时变类别 | 当前外层批循环已完成实体/总实体；不同工作段独立计数，合并时恢复旋转指示器 |
| 线性求解 | 使用首个已完成迭代的残差r0及当前残差r，p=clamp(1-(r-tol)/(r0-tol),0,1)；允许回退，不按迭代上限估算 |
| 总运行 | 成功发布且游标接受的实际时间步/预演总步数；时间表节点插入计入总数，初态不计一步 |

tol=max(atol,rtol*||b||2)。首轮之前显示等待，初值或首轮达标直接显示100%；非有限值时无百分比。
残差注明递推/真实来源；100%不代表输出成功，绿色accepted仅在规定温度投影及真实残差验收通过后出现。
Driver API是CUDA驱动API版本，不是NVIDIA驱动软件包号；Memory*为启动时设备整体已用/总显存，含其他进程，不是本进程峰值。

运行中Q/Esc/Ctrl-C或SIGINT/SIGTERM请求取消。在既有GPU调用及完整迭代求解结束后的安全边界检查，
随后整理摘要并等待计算线程退出，返回130。终端断开同样请求取消；不强行终止CUDA，不提供暂停/重启。
自动模式输出重定向时不输出ANSI控制序列，适合批处理；机器读取应使用结果目录的结构化文件。
当前终端实现及PTY验收范围是Linux远程服务器，不声称已经验证原生Windows控制台。

## 算例配置

配置为严格 UTF-8 JSON，根字段 schemaVersion 必须是整数 1。字段名和枚举值区分大小写；未知字段、重复键、非法数字与错误类型会被拒绝。示例 [steady.json](../example/cases/steady.json) 与 [transient.json](../example/cases/transient.json) 是可直接运行的完整配置。

| 根字段 | 要求 | 内容 |
| --- | --- | --- |
| schemaVersion | 必需 | 当前为 1。 |
| name、description | 可选 | 算例显示名和说明。 |
| mesh | 必需 | 网格来源、格式和坐标缩放。 |
| analysis | 必需 | 稳态/瞬态、二维/三维及活动域。 |
| materials | 必需、非空 | 材料参数定义；id 唯一。 |
| materialAssignments | 必需、非空 | 将活动单元逐一分配给材料。 |
| sources、boundaries | 可选 | 体热源和外边界条件；缺省为空。 |
| boundaryPolicy | 可选 | 未指定边界及自然边界叠加策略。 |
| initialState、time | 瞬态必需 | 初始温度和时间区间。稳态禁止。 |
| discretization、linearSolver、output | 可选 | 离散、线性求解和结果发布选项。 |

### 网格与活动域

~~~json
{
"mesh": {"path": "plate.msh", "format": "msh", "coordinateScale": 0.001},
"analysis": {
  "type": "steady", "space": "planar2d", "thickness": 0.01,
  "domain": {"kind": "physicalGroup", "dimension": 2, "tag": 11}
}
}
~~~

- mesh.path 必需；format 默认 msh，仅接受 msh。读取器支持 MSH 2.2、4.1 的 ASCII 和二进制文件。coordinateScale 默认 1，必须大于 0；缩放后的坐标以米表示，原始网格不被改写。
- analysis.type 为 steady 或 transient；space 为 planar2d 或 solid3d。二维域在全局 XY 平面，必须提供正的 thickness（米）；三维禁止 thickness。geometryTolerance 默认 1e-12 米，用于二维平面检查。
- domain 选出参与求解的同维单元，不能仅凭整个 MSH 文件推断。当前热求解接受二维 Triangle3 和三维 Tet4；不支持的单元不会被悄悄降阶。

区域选择器可用于 domain、材料分配、热源、边界、初始温度覆盖和区域监测。物理组叶节点为 {"kind":"physicalGroup","dimension":2,"tag":11}，也可将 tag 改为 name，二者只能选一个。并集/交集写作 {"kind":"union","items":[选择器,...]} 或 kind=intersection；差集写作 {"kind":"difference","base":选择器,"remove":选择器}。dimension 必须与目标单元的维度一致：三维域通常为 3，三维外表面边界通常为 2；二维域与边界分别通常为 2 和 1。选择结果为空是错误，持久 tag 不等于连续数组下标。

### 材料与热源

常量参数写为 {"kind":"constant","value":300}。瞬态热源、规定温度、外向热流、换热系数和环境温度还可使用时间表；材料和初态仍为常量，空间函数、温度依赖与张量尚不支持。

~~~json
{"kind":"timeTable","interpolation":"linear","outside":"error",
 "points":[{"time":0,"value":300},{"time":3,"value":303}]}
~~~

time为绝对秒数，value沿用该物理字段单位。interpolation必填linear（至少2点）或previous（至少1点）；previous在节点取新值。outside默认error，要求覆盖整个运行区间，也可用hold延续首末值。节点必须有限、严格递增，不会自动排序；温度及换热系数非负，热源与热流可负。完整字段与异常说明见[时间表协议](case.md)。

时间表节点插入原有名义时间网格，maxSteps和输出间隔均按实际步数计。每步使用步末右连续值；跳变处存在离散误差，节点对齐并不精确积分跳变载荷，需要时减小time.step。共享规定温度在全运行区间相容，初始温度须满足起点规定值。可直接运行随项目分发的时变示例：

~~~bash
./build/emberfield example/cases/time-varying.json --output-root ./results --output time-varying
~~~

该示例在同一网格中包含启停体源、升温边界、变化的外向通量、换热系数和环境温度。无需新增CLI参数。

| 位置 | 字段 | 单位和约束 |
| --- | --- | --- |
| materials[] | id、displayName、conductivity | id 必需且唯一；displayName 可选；conductivity 必需且严格正，单位 W/(m K)。 |
| materials[] | density 与 specificHeat | 瞬态热容量的一种表示，两者必须同时给出且严格正；单位分别为 kg/m³、J/(kg K)。 |
| materials[] | volumetricHeatCapacity | 瞬态热容量的另一种表示，严格正，单位 J/(m³ K)；不能与前两项并用。稳态可省略热容量。 |
| materialAssignments[] | id、region、material | 必需；region 选活动单元，material 引用材料 id；每个活动单元必须恰好分配一次。 |
| sources[] | id、region、kind、value | kind 仅 volumetric；value 是 W/m³，可正可负。多个源在交叠单元相加。 |

这里的体热源不是总功率 W；若已知总功率，调用者应按受热体积换算为 W/m³。所有输入采用 SI 单位和绝对温度 K，不自动转换摄氏度。

### 边界条件

boundaries[] 的每项包含唯一 id、外边界 region、kind 及对应参数。边界 region 必须匹配活动域的真实外侧；只给出一个低维物理组并不足以保证其有效。

| kind | 专有字段 | 含义 |
| --- | --- | --- |
| temperature | temperature | 指定温度 K，非负。 |
| outwardFlux | flux | 外向热流密度 W/m²；正值表示热量流出，负值表示流入。 |
| convection | coefficient、ambientTemperature | 换热系数 W/(m² K) 非负；环境绝对温度 K 非负。 |

boundaryPolicy.defaultBoundary 目前仅能为 adiabatic（未指定外侧绝热）；naturalOverlap 默认为 error，也可设为 add，允许同一外侧的自然条件相加。指定温度与自然条件在同一外侧不能混用；共享自由度的指定温度必须相容。二维热流输入仍是 W/m²，积分时程序才计入 thickness。

### 瞬态与求解器

| 字段 | 可选值、默认值与约束 |
| --- | --- |
| initialState.defaultTemperature | 瞬态必需，全域初始绝对温度 K。 |
| initialState.overrides[] | 可选；每项有 id、region、temperature，不同覆盖区域不能重叠。物理初温不同于求解器初猜。 |
| time.start、end、step | 瞬态必需，单位秒；end > start，step > 0，最后一步可缩短。 |
| time.method、maxSteps | method 仅 backwardEuler；maxSteps 默认 1000000，为正整数。 |
| discretization.family、order、quadrature | family 仅 lagrangeContinuous，order 默认 1，quadrature 仅 auto；当前可执行空间为一阶。 |
| linearSolver.method | bicgstab（默认）或 cg；CG 仅应在离散系统满足对称正定前提时使用。 |
| linearSolver.preconditioner | 仅 {"kind":"identity"}；独立求解组件中的 AMG 尚不能通过算例协议选择。 |
| linearSolver.relativeTolerance、absoluteTolerance | 默认分别为 1e-8、1e-12；均非负且至少一个为正。 |
| linearSolver.maxIterations、initialGuess | 默认 1000、zero；瞬态可设 previousStep，以前一接受温度作为数值初猜。 |

### 输出配置

~~~json
{
  "output": {
    "format": "vtu", "encoding": "appendedRaw",
    "fields": ["temperature", "temperatureGradient", "heatFlux"],
    "schedule": {"kind": "everySteps", "interval": 10},
    "includeInitial": true, "includeFinal": true,
    "energy": {"enabled": true, "absoluteToleranceW": 1e-8, "relativeTolerance": 1e-8}
  }
}
~~~

上例的 schedule、includeInitial 仅用于瞬态；稳态应删去它们。output.format 仅 vtu；encoding 为 appendedRaw（默认）或 ascii。fields 必须含 temperature，可额外选 temperatureGradient、heatFlux。overwrite 默认为 false；true 只允许备份并替换有效的 EmberField 旧结果目录，不能任意清空目录。瞬态 schedule 默认为每步，interval 必须为正；includeInitial、includeFinal 默认 true。关闭最终帧不会关闭逐步 CSV；关闭 energy.enabled 后热量字段为空。热量平衡超容差目前记录警告，不会将已收敛解改判为失败。

monitors 可选数组，按 id 唯一标识：node 使用正的 nodeTag；point 使用米制 positionM 三元组及可选 toleranceM；region 使用 region 选择器与 statistics 数组，统计量可为 temperatureMin、temperatureMax、temperatureMean、storedEnergy（仅瞬态）。output.limits 可在算例中设置 maxPoints、maxCells、maxHostBytes、maxDeviceBytes、batchCells、maxFrames、maxMonitors、maxOutputBytes；具体默认值见下节，独立资源文件可覆盖除 batchCells 外的输出预算。

## 性能与资源配置

资源文件是可选的，项目根目录 [limits.json](../limits.json) 是完整示例。它只设各阶段的拒绝阈值，不是总进程内存、显存或磁盘的硬隔离，也不会预分配所写的容量。配置错误在创建结果目录之前报告。

~~~bash
./build/emberfield example/cases/transient.json \
  --resource-limits limits.json --output-root ./results --output larger-run
~~~

文件必须含整数 schemaVersion: 1；其余分组、字段可省略，省略时沿用对应模块默认值。所有预算必须是可表示的正整数字面量，不接受 1.0、1e3、0、字符串或布尔值。未知字段和重复键会被拒绝。output 分组逐字段覆盖算例 output.limits；未写入的字段保留算例值，算例也未给出时才使用默认值。资源文件中的 integration 预算作用于 GPU 积分批；提高阈值本身不会自动扩大内部批大小。

| 分组 | 字段（默认值） | 限制对象 |
| --- | --- | --- |
| case | maxBytes 16777216；maxValues 1000000；maxStringBytes 1048576；maxNesting 128；maxSelectorDepth 64；maxDiagnostics 100 | 算例 JSON 字节、值/绑定记录、字符串、嵌套、选择器深度、诊断保存数。 |
| mesh | maxRecords 100000000；maxBytes 2147483648 | MSH 单数组记录数及输入文件字节。 |
| space | maxCells 1000000；maxLocalDofs 4000000；maxGlobalDofs 4000000；maxDiagnostics 100 | 活动单元、局部/全局自由度和空间诊断。 |
| preparation | maxCells 10000000；maxDofs 10000000；maxFacetUses 40000000；maxAssociations 100000000；maxDiagnostics 100 | 几何与条件准备的单元、自由度、侧访问、关联和诊断。 |
| assembly | maxDofs 10000000；maxStencils 10000000；maxLocalEntries 64000000；maxNonzeros 64000000 | 全局符号组装维度、模板、局部条目与 CSR 条目。 |
| integration | maxElements 65536；maxScalars 16777216 | 单 GPU 积分批的实体和每个平坦数组标量数。 |
| output | maxPoints 4000000；maxCells 1000000；maxHostBytes 1073741824；maxDeviceBytes 536870912；maxFrames 1000000；maxMonitors 1000；maxOutputBytes 107374182400 | 输出点/单元、后处理缓存、帧、监测和文件字节。 |

例如仅提高空间自由度和输出主机缓存预算：

~~~json
{
  "schemaVersion": 1,
  "space": {"maxGlobalDofs": 8000000},
  "output": {"maxHostBytes": 4294967296}
}
~~~

配置不会控制求解器内部或第三方库的全部工作空间。算例中的 output.limits.batchCells 默认 4096，控制后处理设备批大小，不属于 --resource-limits 协议。详细预算合并和错误规则见[运行资源限制](resource-limits.md)。

## 查看结果

run-summary.json 包含有效配置、网格规模、输出清单、计时和终止状态；steps.csv 含每个接受步和失败尝试；启用监测时有 monitors.csv，启用热量诊断时有 boundary.csv；run.log 是人读日志。fields/ 下的 VTU 保存温度及所选派生场，瞬态 temperature.pvd 列出保存帧的物理时间。VTU 的温度单位 K，梯度 K/m，热流 W/m²。完整字段、提交前缀与故障语义见[命令行与结果文件](results.md)。

## CUDA/C++20 公共 API

CLI 是最简运行入口；库的公开头文件在 include/。安装后在消费者项目中查找并链接 CMake 目标：

~~~cmake
cmake_minimum_required(VERSION 3.24)
project(HeatClient LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
find_package(EmberField 0.1 CONFIG REQUIRED)
add_executable(heat-client main.cu)
target_link_libraries(heat-client PRIVATE EmberField::emberfield)
~~~

配置时设置合适的 CMAKE_CUDA_ARCHITECTURES 和 CMAKE_PREFIX_PATH。主要入口如下，参数、异常与对象生命周期的完整契约以对应公开头文件的 Doxygen 注释为准：

| 接口 | 作用与结果 |
| --- | --- |
| emberfield::version() | 查询链接库的语义版本；不初始化 CUDA。 |
| emberfield::io::loadResourceLimits(path) | 读取独立资源策略；协议或 IO 错误抛异常。 |
| emberfield::io::loadCase(path, limits, outputLimits) | 读取并验证 JSON，返回 definition_ 与 diagnostics_；不读取网格、不创建结果目录。 |
| emberfield::io::loadMsh(path, limits) / saveMsh(path, document, encoding, version) | 单独读写 MSH 网格文档；网格本体不存储文件编码。 |
| emberfield::problem::bindCase(definition, mesh, limits) | 把配置的物理组映射到共享只读网格；成功时返回 bound_。 |
| emberfield::problem::buildTemperatureSpace(bound, limits) | 对活动域构建局部到全局温度自由度映射。 |
| emberfield::problem::prepareThermalCase(bound, space, options) | 验证几何和物理条件，成功时返回 prepared()。 |
| emberfield::problem::buildThermalAssemblyPlan(prepared, options) / assembleThermalModel(plan) | 建立符号计划，GPU 积分与全局组装得到热模型。 |
| emberfield::problem::buildThermalSteadySystem(model) / buildThermalStepSystem(model, previous, dt) | 构造已经处理已知温度的稳态或后向 Euler 线性系统。 |
| emberfield::problem::runCase(input, output, interrupted, limits) | 执行完整算例并发布结果；返回与 CLI 一致的业务退出码。 |

若只需要完整运行，main.cu 可以写为：

~~~cpp
#include <IO/Case/ResourceLimits.cuh>
#include <Simulation/Case/Run.cuh>
#include <exception>
#include <iostream>

int main() {
    try {
        auto limits = emberfield::io::loadResourceLimits("limits.json");
        return emberfield::problem::runCase("example/cases/steady.json", "results/plate", nullptr, limits);
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
~~~

如果只想检查输入并取得活动域自由度数，不必组装或求解：

~~~cpp
#include <IO/Case/Case.cuh>
#include <IO/Mesh/Msh.cuh>
#include <Simulation/Case/Binding.cuh>
#include <Simulation/Case/TemperatureSpace.cuh>
#include <exception>
#include <iostream>
#include <memory>
#include <utility>

int main() {
    try {
        auto read = emberfield::io::loadCase("example/cases/steady.json");
        if (!read.definition_) return 2;
        auto document = emberfield::io::loadMsh(read.definition_->mesh_.path_);
        auto mesh = std::make_shared<const emberfield::mesh::Mesh>(std::move(document.mesh_));
        auto binding = emberfield::problem::bindCase(*read.definition_, mesh);
        if (!binding.bound_) return 2;
        auto space = emberfield::problem::buildTemperatureSpace(*binding.bound_);
        if (!space.space_) return 2;
        std::cout << "DOF: " << space.space_->dofCount() << '\n';
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
~~~

上面第二段只用退出码表示预期输入错误；实际程序还应输出 read.diagnostics_、binding.diagnostics_、space.diagnostics_ 里的诊断。完整的组装和求解调用见 [solve-case.cu](../example/Simulation/solve-case.cu)。读取、绑定、空间构建各自只发布完整成功对象；调用者应先检查结果再访问。程序化构造 CaseDefinition 时先调用 validateCase，不能把解析成功视为几何、离散和求解能力已经全部得到证明。

性能分析使用可选的[NVTX作用域标签](profiling.md)，其中说明构建开关、公共宏与Nsight Systems采集方式。
