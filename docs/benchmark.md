# 大规模算例与Benchmark

Benchmark输入必须是完整可运行的热算例：网格、材料、边界、热源、时间设置及求解精度共同构成工作负载。自由度多不必然说明计算负载足够，记录时还需报告活动单元、CSR非零元、实际时间步、迭代次数、各阶段耗时与内存占用。

## 准备大网格

`scripts/benchmark/prepare.py`接受已可运行的原生算例，以Gmsh一致细化扩大网格。准备环境需要Python 3.10或更新版本、Gmsh Python包及NumPy；求解器本身不增加这些依赖。建议将工具装在独立虚拟环境。当前核对工具版本为Gmsh 4.15.2和NumPy 2.5.4。

~~~bash
python scripts/benchmark/prepare.py /path/to/source/case.json \
  --output data/benchmark/plate --levels 1 2 --steps 20 \
  --without-monitors
~~~

| 参数 | 含义 |
| --- | --- |
| case | 已可运行的原生算例；网格相对配置解析。 |
| --output | 新建输入包根目录；必须不存在，避免覆盖既有输入。 |
| --levels | 保存的细化层号，0表示原网格；Triangle3每层分为4个三角形，Tet4每层分为8个四面体。 |
| --steps | 可选的瞬态名义步数；保持原起止时间，以区间长度除以步数得到dt。时间表事件仍可能增加实际步数，不等于最终实际步数。 |
| --relative-tolerance | 可选地显式替换相对残差阈值；默认保留原值。比较优化前后必须保持精度一致。 |
| --absolute-tolerance | 可选地显式替换绝对残差阈值；默认保留原值。最大迭代数继续来自原算例。 |
| --threads | Gmsh线程设置，默认4；重复生成的节点顺序不保证字节一致。 |
| --without-monitors | 显式清空可选点/区域监测，不改变热平衡开关及物理条件；默认保留。 |
| --reuse-from | 已生成输入包的根目录；复用对应level-N网格，以来源哈希、细化层号和文件哈希核对。适合同网格比较恒定和时变条件，避免重新生成造成编号差异。 |

每档生成`mesh.msh`、`case.json`、`limits.json`与`manifest.json`，根目录保留`source-case.json`。manifest记录来源路径和SHA256、原始网格哈希、Gmsh版本、细化层数、节点与单元计数、每个物理组的名称和单元覆盖、显式配置变化及最终文件哈希。准备失败可能留下部分目录；只有完整manifest对应的档位可用于后续验证，且仍须通过正式程序加载求解。

输入包保存MSH 4.1二进制网格。工具要求活动维度只有一阶Triangle3或Tet4，且保存节点全部属于该维度；不自动丢弃游离节点、不自动降阶。活动域选择器仍由原配置决定，最终DOF/nnz以求解摘要为准。每次细化检查物理组集合、名称及单元数按预期增长。相邻实体共享节点由Gmsh统一细化，不复制独立小模型凑大问题。

细化作用于已有直边离散网格，不重新投影CAD曲面；它是固定离散几何上的规模实验，不自动构成CAD几何收敛验证。工具不生成新材料参数、不将稳态擅自改为瞬态、不把初态改为精确解。相对容差改变和监测移除均记录于manifest。

默认保留原算例的线性求解精度。相对残差小不一定表示温度增量或热平衡误差足够小：瞬态系统的质量项与绝对温度可能使右端项远大于加热增量。尤其不能为缩短计时统一放宽到某个相对容差；显式改变精度后须重新检查热平衡和结果，出现告警的候选输入不作为已通过数值验收的基准。运行器的completed表示程序完整执行，热平衡状态须另外核对。

## 工作负载选择

| 类型 | 适用负载 | 应关注的阶段 |
| --- | --- | --- |
| 大规模二维板 | 百万级到数百万DOF、不规则域 | 网格输入、拓扑和符号准备、稀疏组装、低阶CSR带宽 |
| 多材料三维器件 | 数十万到百万DOF、数百万Tet4、薄层和导热率反差 | 几何准备、体与边界积分、组装、迭代求解和数据搬移 |
| 固定条件瞬态 | 重复系统构造、矩阵版本可复用 | 每步矩阵组合、历史项、setup缓存、求解和后处理 |
| 时变条件瞬态 | 改变源、通量、规定温度、h和环境温度 | 分类重组装、事件对齐、矩阵版本变化与setup重建 |

本项目既有公开来源可作为细化输入：ElmerCSC `heateq_bdf2`的L形域热传导适配，以及femx公开环形加热器的八体区域热传导适配。前者改为Triangle3与后向Euler；后者使用给定均匀体源替代原电热耦合。它们是公开数值模型的适配，不是实验或工业测量复现。

来源入口：[Elmer原算例](https://github.com/ElmerCSC/elmerfem/tree/56b16bd49564d5b16fdc57862ccda024043b0f39/fem/tests/heateq_bdf2)、[femx固定版本](https://github.com/yspkm/femx/tree/42f6b9cbe74a1b5cf31b3962b1b1b69f88297fff)。源数据与许可证按各自来源保存，不因EmberField使用MIT而改变其许可；大网格、结果及第三方数据放在被忽略的data目录，不直接随代码发布。

## 资源配置

准备器按实际网格规模生成分阶段预算，覆盖算例绑定访问、MSH读取、空间、几何准备、符号展开与后处理。后处理预算按照当前实现的保守估计留有余量，不等于实测显存占用；提高预算不会预分配该数量的内存，也不保证实际主机内存足够。

当前区域监测的选择操作使用独立默认记录预算，可能在百万单元区域上失败，即使主算例绑定预算已扩大。纯计算基准可显式使用`--without-monitors`；不得把这种配置当作区域监测功能的性能结果。需要监测的工作负载应保留该选项并单独验证其可用规模。

## 运行与统计

正式性能测量采用Release、本机GPU架构及关闭NVTX的独立构建。示例架构86对应RTX A6000，其他GPU应按实际架构设置。性能分析采集使用独立NVTX构建，采集耗时不能直接作为无分析器的基准成绩。

~~~bash
cmake -S . -B build-benchmark -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DEMBERFIELD_ENABLE_NVTX=OFF \
  -DBUILD_TESTING=OFF -DEMBERFIELD_BUILD_EXAMPLES=OFF
cmake --build build-benchmark --parallel 4

python scripts/benchmark/run.py build-benchmark/emberfield \
  data/benchmark/plate/level-1/case.json \
  --resource-limits data/benchmark/plate/level-1/limits.json \
  --output data/benchmark/results/plate-baseline --warmup 1 --repeat 3
~~~

运行器只需要Python标准库。各次运行串行，预热与正式样本分开，默认1次预热、3次测量。结果目录必须不存在，所有运行都保留原生输出、完整终端日志及可用的GNU time资源记录。`--warmup 0 --repeat 1`只用于确认大输入能跑通，不称为稳定的统计benchmark。

`benchmark.json`保存输入/网格/资源配置/可执行文件哈希、当前源码树版本、可识别的CMake构建配置、主机信息，以及每次运行的命令、进程墙钟、程序阶段计时、DOF/nnz/活动单元、实际步数、总迭代数、最大接受残差和热平衡状态。Linux有GNU time时另外保存进程最大RSS；这不是显存峰值。源码树提交不等于编译版本证明，可执行文件哈希是比较程序身份的直接依据。

仅对全部完整成功的正式运行计算中位数、均值、极值和样本标准差。任一次非零退出或摘要不是completed，则报告failed并保留原始证据，不将失败时间当作性能成绩。摘要中的statisticalBenchmark仅表示完成至少一次预热、三次测量的次数门槛，不表示统计显著或环境独占。

输出统一保留温度末帧，避免输出频率随实际步数变化；仍计入实际I/O和后处理阶段。没有禁用数值验收、减少网格连接或放宽失败判定。热平衡状态原样记录；它与线性残差含义不同，须结合源算例的容差解释。

比较前后保持输入哈希、精度、实际步数、输出策略、GPU、构建架构及批参数一致；避免同时运行其他GPU负载，并记录共享服务器状态。端到端总耗时、各阶段耗时和单独求解耗时分别报告，不能将某阶段加速倍数表述为整个程序的加速倍数。
