# 几何、局部热运算与条件准备

第4—6步在完整配置、网格绑定和连续P1空间之上，提供米制仿射几何、数值积分、独立热运算以及无冲突物理条件。当前支持XY平面Triangle3和三维Tet4、常量标量物性、固定网格。网格模型的表达范围仍大于热离散后端，不自动降阶未知单元。

## 完整配置入口

~~~cpp
auto input = thermofem::io::loadCase(configPath);
auto document = thermofem::io::loadMsh(input.definition_->mesh_.path_);
auto binding = thermofem::problem::bindCase(
    *input.definition_, std::make_shared<const thermofem::mesh::Mesh>(std::move(document.mesh_)));
auto space = thermofem::problem::buildTemperatureSpace(*binding.bound_);
auto result = thermofem::problem::prepareThermalCase(binding.bound_, space.space_);
~~~

每一步必须先检查成功对象及诊断，不能直接解引用失败结果。完整受检用法见example/prepare-inspect.cu。PreparationResult::ok仅代表本阶段成功；旧BoundCase能力报告不变，尚未生成A、b，也不认证稳态纯Neumann、全局正定性或CG适用性。

~~~sh
./build/example/prepare-inspect example/cases/steady.json cpu
./build/example/prepare-inspect example/cases/transient.json gpu
~~~

gpu模式批量执行局部积分，并显式回读与主机路径比较，用于验证；不是性能基准。默认批次4096个实体，不永久缓存全部单元矩阵。

## 分层与单位

fem::buildPreparedGeometry只读取Mesh/space和GeometryOptions，按原始坐标乘coordinateScale一次得到米制几何。二维容差也在米制坐标上判断；原Mesh坐标不变。负det表示局部节点方向，保留原序，积分使用正测度，梯度使用带符号J的逆转置。

evaluateCellIntegration与evaluateFacetIntegration返回拥有数组的实际积分表，权重已经包含Jacobian但不含厚度。独立thermal::integrateConductivity、integrateCapacity、integrateVolumeSource、integrateOutwardFlux、integrateConvection接收按该表积分点顺序采样的系数。样本长度必须完全匹配，不隐式广播。

sectionScale二维为米制厚度，三维必须1。K/H单位W/K，M单位J/K，载荷单位W；M不是M/dt。q·n=g>0代表向外散热，其右端载荷为负。对流同时产生H与f∞，不能只保留其中一个。auto使用二次精确规则得到一致容量矩阵，不做一点欠积分或质量集中。

体项按母元局部DOF顺序，侧项按parentLocalDofs/有序迹DOF顺序，不是母元前nf项。实际矩阵装配及温度消元是后续模块职责。

## 条件和初态

PreparedConditions组合材料索引、逐单元体源关联、逐侧自然条件、唯一温度约束及瞬态全空间初温。多个源相加，同源选择去重。NaturalOverlap::Add只允许自然项叠加，不允许温度与自然条件同侧冲突；不同侧在角点相交的自然贡献仍保留。

共享DOF的规定温度必须相同。当前常量字段规范化±0后精确比较，不设置隐含浮点容差。初温按每单元override或default向节点提出值，共享节点不同则拒绝，不按覆盖顺序改值、不平均。初温还必须与t0规定温度相容；物理初温不同于solver.initialGuess。

## GPU批处理

热局部积分有可复用GPU设备表：uploadCellBatch/uploadFacetBatch上传同space、同维数、同局部维数及同规则表；integrate*Batch批量计算，返回CudaBuffer组合的设备矩阵/向量，供[全局GPU组装](assembly-system.md)直接消费。数据以实体为首维、矩阵行主序，无原子散加。每个kernel处理整个批次，不逐单元启动。准备/拓扑/冲突检查保留主机端。

problem::evaluateCellBatch按指定活动行返回GPU K、瞬态M和按定义序累加的源向量；evaluateBoundaryBatch按自然项行处理同类型通量或对流。侧Add多项仍分别计算，不合并h与T∞。调用者按GpuBatchLimits分批，不能要求将任意大网格一次上传；保留批次行映射用于装配。

独立上传的设备表可多次复用；算例便利入口每批创建/上传，不承诺跨调用缓存。数值结果默认留在设备，copyToHost是显式回读。每次积分会同步回读O(批实体数)状态，以保证范围失败不发布半成品；当前不是全异步多流流水。设备归属必须匹配当前设备，同尺寸异来源向量不能相加。

主机参考使用扩展精度中间乘积，GPU使用double及指数分解乘积防止不必要的中间溢出；结果按误差容限对照，不承诺逐位相等或对所有极端数值给出完全相同失败边界。无性能加速倍数承诺。

## 数值失败与资源

invalid_argument表示非法策略、样本形状/范围、精度或厚度；out_of_range表示行或侧越界；EvaluationError提供稳定数值代码及母元/局部侧等上下文。准备入口只转换明确数值失败为限长诊断，不吞没bad_alloc或CUDA资源错误。失败不修改旧输入，不发布部分准备对象或部分局部结果。

计数预算限制单元、DOF、母侧use和条件展开访问；不是STL堆内存硬字节上限。几何局部邻接检查不能替代全域相交检测，共形无悬挂仍为网格前提。高阶、张量物性、非线性、周期/MPC、辐射、任意嵌入曲面及轴对称不在当前后端范围内。
