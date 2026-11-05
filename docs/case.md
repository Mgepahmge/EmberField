# 算例输入与网格绑定

## 分层接口

`problem::CaseDefinition`是与JSON、Mesh和CUDA无关的宿主值模型。`validateCase`校验程序化定义；`io::readCase/loadCase`负责严格配置协议；`problem::bindCase`接受定义与`shared_ptr<const mesh::Mesh>`，产生不可变`BoundCase`。调用方不得通过其他可变别名修改该Mesh。

绑定拥有定义快照及网格共享所有权，单元引用使用块索引与块内索引，持久标签不充当数组下标。坐标缩放由`BoundCase::position`统一执行，不原地修改Mesh。求解器缓存、矩阵、初始猜测和时间状态不进入网格模型。

## JSON协议

配置为UTF-8 JSON，`schemaVersion`必填整数1；字段和枚举区分大小写。拒绝重复键（含转义后同名）、未知字段、非整数标签和非法数字语法。所有JSON整数字面量须在int64范围内，越界不回退浮点。默认限额为16MiB、1000000值、1MiB单字符串、128层JSON、64层选择器、100条诊断；调用者可通过`CaseLimits`调整，限额不能为0。

原生算例快速运行指南见[原生算例示例](native-case-example.md)。完整示例为`example/cases/steady.json`和`transient.json`，其网格为同目录`plate.msh`。相对网格/输出路径基于配置文件目录，不执行环境变量和命令替换。

| 根字段 | 语义及默认 |
| --- | --- |
| name、description | 可选显示信息 |
| mesh | path必填；format默认msh；coordinateScale默认1，必须严格正 |
| analysis | type为steady/transient、space为planar2d/solid3d、domain必填；二维thickness必填且正；三维禁止厚度；geometryTolerance默认1e-12m |
| materials | 非空数组；id唯一、conductivity必填且正；displayName可选 |
| materialAssignments | 非空数组；每项id、region、material必填；活动单元必须恰好分配一次 |
| sources | 默认空；每项id、region、kind=volumetric、value；不同源在交叠区域相加 |
| boundaries | 默认空；每项id、region、kind和专有参数；类型见下 |
| boundaryPolicy | defaultBoundary仅adiabatic；naturalOverlap为error（默认）或add |
| initialState | 仅瞬态且必填，defaultTemperature必填；overrides默认空，每项id、region、temperature |
| time | 仅瞬态且必填，start/end/step；end>start，step>0；method仅backwardEuler；maxSteps默认1000000 |
| discretization | family仅lagrangeContinuous，order默认1且正，quadrature仅auto |
| linearSolver | method默认bicgstab，也接受cg；预条件子仅identity；relativeTolerance默认1e-8，absoluteTolerance默认1e-12，maxIterations默认1000；initialGuess默认zero，瞬态可用previousStep |
| output | format为vtu；默认temperature，可选梯度/热流、编码、监测、热量与资源预算，详见[结果输出](results.md)；CLI输出路径覆盖directory；overwrite默认false；瞬态schedule默认每步，includeInitial/includeFinal默认true |

显式整数选项不接受`1.0`或`1e0`；布尔不接受0/1。常量字段写为`{"kind":"constant","value":300}`。瞬态体源与边界还可使用下述时间表；材料和初态保持常量。尚不支持张量、温度依赖和空间表达式。

### 时间表

~~~json
{"kind":"timeTable","interpolation":"linear","outside":"error",
 "points":[{"time":0,"value":300},{"time":2,"value":320}]}
~~~

| 字段 | 类型、默认值与约束 |
| --- | --- |
| kind | 必填字符串timeTable，与constant互斥 |
| interpolation | 必填linear或previous；线性至少2点，previous至少1点 |
| outside | error（默认）拒绝外推；hold延续首末值 |
| points | 必填有序非空数组，不自动排序、不合并节点 |
| points[].time | 有限绝对物理时间，单位s；转换成double后必须严格递增 |
| points[].value | 有限物理值，单位由父字段决定 |

允许的父字段为sources[].value、temperature边界的temperature、outwardFlux边界的flux、convection边界的coefficient和ambientTemperature。稳态、材料物性、初始温度禁止时间表。温度和环境温度全表非负，换热系数全表非负，体源与外向通量允许负值。error表必须覆盖time.start至time.end；hold允许单侧或全部运行区间处于表外，仍检查全部节点物理范围。线性插值与保持不会越过节点值范围。

linear在相邻节点之间插值；previous在区间[t_i,t_{i+1})取v_i，精确落在节点t_i时取新值v_i。time是绝对时间，不减去time.start。所有区间内表节点自动插入名义网格start+k*step，不改变后续名义节点；重复时刻合并，不生成零时间步。maxSteps限制插入事件后的实际步数，运行前预检；输出everySteps按实际步号计数。

全部外载荷与规定温度采用后向Euler的步末右连续值，包括跳变节点。事件对齐不等于精确积分脉冲：例如Q在t=1跳变，新值也用于终点为1的这一步，有限步长下会存在时间离散误差。需要更高时间精度时缩小step。h和环境温度分别插值，再计算h*T∞，不对端点乘积直接插值；乘积溢出时终止并记录失败尝试。

共享温度DOF上的多条约束须在整个闭区间相容，而非仅初态相同；比较联合分段节点和真实左极限。相容性采用精确数值比较，不使用近似容差合并边界。初态仍须满足g(time.start)。完整可运行示例见example/cases/time-varying.json。

材料热容量采用density与specificHeat同时提供，或者volumetricHeatCapacity单独提供；三项均须严格正，两种方式互斥。稳态允许缺省全部热容量，瞬态所有材料必须具备热容量。密度与比热乘积不得溢出。

## 单位与物理语义

内部固定SI：坐标m、时间s、绝对温度K、导热率W/(m K)、密度kg/m³、比热J/(kg K)、体热容J/(m³ K)、体源W/m³、热流W/m²、换热系数W/(m² K)。温度必须非负，不将20自动理解为摄氏度。所有浮点量须有限。

二维为全局XY平面内的平面传热，活动节点缩放后z应在容差内为0，厚度以米输入。任意三维曲面不等于二维平面；轴对称、表面导热和杆模型尚不支持。

外向热流定义为q·n，其中q=-K grad(T)，正号为热量流出。边界类型：temperature使用temperature；outwardFlux使用flux（可正可负）；convection使用非负coefficient与非负ambientTemperature。输入总功率W不能冒充体源W/m³。

## 区域选择与边界

物理组选择为`{"kind":"physicalGroup","dimension":2,"tag":11}`，或将tag替换为name，二者必须恰选一种。标签是维数限定的有符号正整数；名称在同维必须唯一。组成员可以无名；只有名字没有单元会报告空选择。

并交分别使用kind=union/intersection与非空items数组；差集使用kind=difference、base和remove。集合去重且同维，空最终结果错误。块级显式分组覆盖实体分组，显式空表禁止重新继承。

材料、源、初温覆盖不得选中活动域外单元。重复材料分配即使引用同一材料也错误；不同体源可以叠加。未使用材料为警告。初温区域覆盖不得相互重叠，初温与数值初猜分离。

外边界由活动单元侧判断，不能把所有低维物理组当外边界。独立`buildFacetTopology`支持完整一阶Triangle3/Quad4/Tet4/Hex8/Prism6/Pyramid5，按完整节点身份匹配，不按坐标合并。高阶/自定义布局留下阻断项，不静默降阶。低维单元必须与活动外侧匹配；内部/孤立侧拒绝，非流形域拒绝。

自然边界同侧叠加默认拒绝，显式add允许多个自然条件相加；同侧温度与自然条件永远不允许混用。全区间相容的温度约束可共用同侧；不同侧交汇时，在条件准备阶段核对共享DOF的温度曲线与起点初温。未覆盖外侧默认绝热，拓扑完整时报告覆盖摘要。

首期无法证明多分区/幽灵文件全域完整性，保守拒绝；partitionCount为0/1且没有幽灵与非1分区成员时可绑定，仍需后续系统检查。Mesh周期元数据不自动施加温度周期条件。

## 错误与能力边界

预期输入错误通过`Diagnostics`返回，含code、path、对象id、可选网格tag、中文消息/建议与来源位置。路径为JSON Pointer；文件来源列号按UTF-8字节计。静态语义错误可聚合；语法/类型转换遇到首个错误停止。诊断限长仍保留错误状态和省略数。

配置失败无definition，绑定失败无bound；已有输入与旧绑定不变。系统分配异常向上传递，零限额或非绝对流读取基目录抛invalid_argument。普通IO失败返回IO_ERROR；调用方自行给流启用异常时可能抛ios_base::failure。输出目录不在加载阶段创建或清空。

成功绑定不等于可执行。高阶温度空间或非首期域单元的后端限制放在独立capability报告；几何、DOF、可解性、组装、结果输出保留pending项目，`executable()`据此为false；这个报告描述绑定阶段，不随后续处理改变。完整离散与GPU组装调用见[全局组装与线性系统](assembly-system.md)，solve-case示例按配置创建CG/BiCGStab并进行时间推进；VTU输出器另属独立职责。CG的SPD数学前提、每个连通分量的纯Neumann问题及未来AMG适配不得被配置解析替代。

解析依赖为项目内picojson原样副本，见`third-party/README.md`与其原始BSD-2-Clause许可证。公开头文件不暴露第三方类型，安装消费者无需JSON包。所有者指定副本的精确版本未核实，以SHA256定位，不宣称使用某个发布版本。
