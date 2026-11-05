# 命令行与结果文件

瞬态运行在run-summary.json的input.timeEvaluation记录时间采样协议；稳态为null。

| 字段 | 类型与含义 |
| --- | --- |
| sampling | 字符串backwardEulerRightEndpoint，步末右连续值 |
| eventAlignment | 字符串tableKnots，合并并插入所有运行区间内时间表节点 |
| eventCount | 十进制整数字符串，去重后的内部节点数，不含起止点 |
| nominalStepS | 数字，配置名义步长，单位s |
| plannedStepCount | 十进制整数字符串，CLI预检的实际步数；未提供调度计数的库会话为null |

effectiveCase保留完整有序时间表；steps.csv的step、time_s、dt_s以及VTU/PVD时间均为实际事件对齐后的状态。
采样协议与预检计数同时写入run.log。跳变条件的功率和规定温度反力采用离散步区间的解释，不能当作连续时间脉冲的精确积分。

## 运行算例

```bash
emberfield case.json
emberfield case.json --output-root /path/to/results --output plate
emberfield case.json --resource-limits limits.json
emberfield --help
emberfield --version
```

必需位置参数是算例JSON路径。`--output-root`默认是操作系统查询的实际可执行文件目录，
相对路径按启动工作目录解释；`--output`是一个目录名，默认取配置文件名去掉最后的扩展名。
例如 `plate.v2.json` 默认在可执行文件目录下生成 `plate.v2/`。这两个参数组合为最终目录，
覆盖算例中的`output.directory`；求解方法、时间参数、输出内容和覆盖策略仍由配置决定。
`--output`禁止路径分隔符、`..`、控制字符及系统保留名字。

`--resource-limits`指定统一的资源限制JSON文件。项目根目录提供完整的`limits.json`示例；
程序不自动加载它。文件可分别设置读取、空间、准备、组装、GPU积分及输出预算，
省略的字段保留原值。输出预算优先于算例对应字段，其他输出设置仍由算例控制。
字段、默认值、合并顺序与限制范围见[运行资源限制](resource-limits.md)。

退出码：0成功；2配置、绑定或离散准备失败；3求解不收敛或接受残差失败；
1资源、后处理或文件错误；130在步边界响应用户中断。参数语法错误使用CLI11退出码；
资源配置语法或协议错误返回1，且不创建结果目录。
帮助及版本查询不初始化CUDA。SIGINT/SIGTERM只设置标志，不在信号处理器中访问CUDA或文件。

## 输出设置

`output`兼容既有配置；所有对象继续拒绝未知键、重复键和错误类型。

| 字段 | 默认值 | 约束 |
| --- | --- | --- |
| format | vtu | 仅vtu |
| encoding | appendedRaw | appendedRaw或ascii，无压缩 |
| fields | [temperature] | 必含temperature；可加temperatureGradient、heatFlux；不重复 |
| overwrite | false | 非空目录默认拒绝；true仅允许备份有效EmberField旧结果目录 |
| schedule | everySteps、interval=1 | 仅瞬态，正步数间隔 |
| includeInitial | true | 仅瞬态，是否保存初态VTU |
| includeFinal | true | 稳态必须true；瞬态终态强制保存与间隔命中取并集 |
| energy.enabled | true | 每步计算离散热量诊断，不改变求解判据 |
| energy.absoluteToleranceW | 1e-8 | 有限非负 |
| energy.relativeTolerance | 1e-8 | 有限非负 |
| monitors | [] | id唯一，静态定位在求解前检查 |
| limits.maxPoints | 4000000 | 活动输出点上限 |
| limits.maxCells | 1000000 | 活动输出单元上限 |
| limits.maxHostBytes | 1073741824 | 后处理增量数组的保守预算，非进程RSS硬限制 |
| limits.maxDeviceBytes | 536870912 | 后处理增量缓存与SpMV临时副本的保守预算 |
| limits.batchCells | 4096 | 设备运算批大小 |
| limits.maxFrames | 1000000 | 包括初态和强制终帧 |
| limits.maxMonitors | 1000 | 监测条目上限 |
| limits.maxOutputBytes | 107374182400 | 当前会话文件逻辑字节预算，不含旧备份 |

温度、梯度、热流的单位分别为K、K/m、W/m²。二维热流第三分量为0，
热流密度不乘厚度；体积和功率积分包含厚度。当前空间是Triangle3/Tet4上的连续P1，
VTU保持活动单元的局部顺序，不将高阶单元静默降阶。

监测例子：

```json
[
  {"id":"node","kind":"node","nodeTag":"9007199254740993"},
  {"id":"sensor","kind":"point","positionM":[0.2,0.3,0],"toleranceM":1e-10},
  {"id":"body","kind":"region","region":{"kind":"physicalGroup","dimension":2,"tag":11},
   "statistics":["temperatureMin","temperatureMax","temperatureMean","storedEnergy"]}
]
```

节点必须位于活动域。点坐标以米为单位，只在活动域闭包及给定定位容差内插值；
超出容差拒绝，不做任意最近点替代。多个单元命中时选择最小活动单元行。
区域选择器与算例协议一致，取非空活动交集；温度均值按体积积分，storedEnergy仅瞬态合法。

## 文件集合

| 文件 | 内容 |
| --- | --- |
| run-summary.json | schemaVersion=1、程序版本、来源、有效配置、网格规模、身份字典、场清单、表格有效前缀、计时与终止状态 |
| steps.csv | 初态、每个接受步及失败尝试的收敛、真实接受残差、温度范围、能量和帧路径 |
| monitors.csv | 配置监测时生成；每个状态按id及统计量顺序的长表 |
| boundary.csv | 能量诊断启用时生成；逐自然条件向外功率及全部规定温度DOF的总反力功率 |
| run.log | UTF-8事件和诊断，机器数值读取使用JSON/CSV |
| fields/state_0000000000.vtu | 活动域网格及所选场，步号十进制至少十位 |
| temperature.pvd | 瞬态且至少保存一帧时的VTU时间索引 |

稳态只有step=0且无伪物理时间、无PVD。瞬态初态step=0，推进从1开始，实际末步可缩短。
VTU频率不影响每步CSV和监测。失败候选不作为普通温度场发布。

VTU使用`UnstructuredGrid`、小端、UInt64块头；默认raw追加块是二进制载荷，
不能把整个文件交给普通XML文本解析器。可直接在ParaView中打开VTU或PVD。
温度在PointData，梯度和热流在CellData。来源节点tag、单元tag、DOF、活动行、材料、实体分类
及完整物理组CSR分别保存，不能把tag当连续索引。摘要中所有标签、计数、步号使用十进制字符串，
避免JSON消费者通过double损失精度；不可用数值是null，CSV中为空。

## 热量诊断

基于未消元方程`M dT/dt + S T = f`计算；反力是唯一规定温度DOF集合上的负残差和。
不同命名温度边界共享DOF时没有唯一功率分配，因此只输出整体反力，避免重复计数。
平衡误差为“储能变化率 + 规定外向通量 + 对流外向功率 + 温度反力外向功率 - 体热源功率”。
初态只报告储能，不虚构动态平衡；关闭诊断时热量字段为空。超容差写warning，
不把已经满足线性收敛和接受检查的解改判成失败。

## 发布和故障

操作前先获取结果目录旁的独占锁目录（后缀为 .emberfield-lock），已有锁拒绝并发运行。
显式覆盖将整个旧目录重命名为同级备份目录（后缀包含 backup 和运行ID），不递归删除任意非空目录。
输出路径禁止符号链接/重解析点，也不能包含输入配置或网格。
进程异常退出可能留下锁；确认没有运行中的写入者后才能人工处理，不自动抢占。

VTU先写临时文件并关闭后重命名，随后更新PVD、追加完整CSV记录并flush，最后替换摘要。
`lastAcceptedStep`与`lastPersistedStep`分别表示数值接受和结果提交，可能不同。
CSV只将摘要中`committedBytes`和`committedRows`指定的前缀视为有效。
异常时尽力回退表格及PVD；孤立文件不属于摘要清单。目录内容不是跨文件原子事务，
也不保证断电级持久性；强杀或断电后仍可能保留running摘要。不得据此自动续写。

PVD和摘要随每帧完整重写，总开销随帧数可能为二次增长，应为长时程设置合理间隔。
`maxFrames`和字节预算用于拒绝过量输出。VTK/ParaView仅作用户查看及独立验收工具，
不属于程序构建依赖。
