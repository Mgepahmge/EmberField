# 全局稀疏组装与最终线性系统

## 分层接口

assembly::buildAssemblyPlan仅接收全局维度和有序局部DOF模板，不依赖Mesh、Case或JSON。
计划拥有规范int64 CSR图、局部条目槽映射和编号共现连通分量；不调用CUDA。
AssemblyAccumulator把设备局部矩阵与向量散加到该图，输出设备矩阵和载荷。

systems提供独立的稳态复制、后向Euler组合、对称Dirichlet消元及温度场验证/投影。
problem::buildThermalAssemblyPlan、assembleThermalModel只负责将既有准备对象、
GPU局部积分与上述通用运算组合，不在算例层重写数值内核。

~~~cpp
auto plan = thermofem::problem::buildThermalAssemblyPlan(prepared);
auto model = thermofem::problem::assembleThermalModel(plan);
auto system = thermofem::problem::buildThermalSteadySystem(model);
// system.matrix()和system.rhs()可直接交给已有Cg/BiCgStab。
~~~

prepared必须来自成功的prepareThermalCase结果，完整配置读取与诊断见
[几何、局部热运算与条件准备](thermal-preparation.md)。

## 规范图与数值组装

所有原始DOF保留，规定温度不重新编号。每行列索引严格升序、无重复，
所有行预留对角，显式零不会被删除；CSR偏移与列均使用int64。
体模板按活动行排列，之后为自然条件关联行；侧模板按有序迹DOF排列。
不同自然定义可作用于相同侧，保留独立模板，不能先合并h与环境温度。

局部矩阵按实体优先、实体内行主序平铺；局部向量按实体优先、局部DOF顺序平铺。
addMatrix/addVector接收连续模板切片及数量精确匹配的global设备缓冲区。
映射在主机按目标槽稳定分段，每个GPU线程顺序累加一个目标，
避免浮点原子调度改变求和顺序；输入顺序及调用顺序相同时结果可复现。
单元矩阵镜像相等时保持全局数值对称。

每次先计算候选并同步检查有限性，再提交到累加器。
参数或数值校验失败保留旧值；CUDA执行故障后应丢弃受影响累加器。
完整热模型只在全部体和侧批次成功后发布，原准备对象不变。

AssemblyLimits约束全局DOF、模板、全部局部方阵条目和规范nnz。
默认分别为10000000、10000000、64000000和64000000。
ThermalAssemblyOptions默认每批4096实体，并按GpuBatchLimits自动进一步分批。
这些是计数预算，不是精确堆/显存总字节上限。符号排序复杂度O((N+L)log(N+L))，
持久计划O(N+nnz+L)，没有全局N平方数组；数值批映射排序O(Lb log Lb)。
映射可以复用，当前每次add仍生成该切片的分段上传表。

## 物理系统与规定温度

模型保存未约束的S=K+H、瞬态一致容量M及完整载荷f；单位分别为W/K、J/K、W。
稳态A0=S、b0=f。后向Euler对实际步长dt使用

\f[A_0=S+M/dt,\qquad b_0=f+(M/dt)T^n.\f]

S、M必须具有完全相同的规范模式；组合在GPU执行，历史乘积使用cuSPARSE。
不能先清除容量矩阵中的已知温度列再计算历史项。模型可以复用于不同dt，
物理上一时刻温度与求解器数值初猜各自独立。

对于全部规定值D,g，同时从原A0计算自由行
b_i=b0_i-Σ(j∈D)A0_ij*g_j，并将这些列清零；
规定行清零后对角设1、右端设g_i。
保留完整CSR图，输出对称，空D与全D均合法，输入不变。

buildThermalStepSystem要求完整历史温度有限非负且精确满足常值温度约束。
迭代解可经systems::projectTemperatureState在GPU副本中精确恢复规定值，
再用作下一时刻物理状态；这不是把历史向量替换为零初猜。

## 分量诊断

checkThermalSolvability按模板DOF共现关系检查每个分量。稳态分量至少含一个
规定温度DOF或正换热系数侧；h=0无锚定作用。瞬态正容量提供时间项锚定。
未锚定稳态分量区分净载荷不平衡（不相容）与平衡（常数不唯一），两种情况均由
热系统工厂明确拒绝，不自动固定节点。报告包含净功率、绝对功率和容差内兼容性。

上述判断使用已准备热模型的正导热、正测度、连续P1等前提，
不是通用CSR正定性或浮点条件数认证。通用systems接口不替调用者判断CG前提。

## 时间与执行

nextTimeStep不建立无界时间数组，到终点返回空；最后一步使用实际剩余时间。
合并8倍机器精度时间尺度内的终点舍入余步；无法严格前进、步数超额或非法时间参数报错。
只在本步求解成功后推进时间与物理历史。

~~~sh
./build/example/solve-case example/cases/steady.json
./build/example/solve-case example/cases/transient.json
~~~

示例读取完整JSON与MSH、组装全局A,b、按配置选择CG或BiCGStab与恒等预条件，
输出维数、nnz、每步迭代次数、真实残差及最终温度范围。
瞬态支持zero/previousStep初猜，并投影规定温度后传递物理历史。
示例不消费配置中的VTU输出意图，结果文件导出属于独立输出模块。

## 设备、失败与扩展

数值输入为当前同一设备的global存储，不隐式跨设备复制。
参数/模式错误invalid_argument；越界out_of_range；计数超限length_error；
系数组合、历史项、消元等数值范围失败overflow_error；
未锚定热稳态domain_error；CUDA与分配异常直接传播。
M/dt中的非零条目下溢为零也会拒绝，避免静默丢掉时间系数。
通用系统每次生成独立候选，失败不修改原S/M/f或历史。

当前物理后端为Triangle3/Tet4的仿射连续P1、常量标量物性与固定网格，
空间的既有共形无悬挂前提继续适用。符号模板和通用CSR运算独立于这些选择，
可以由未来高阶、非线性或其他物理适配层复用。
