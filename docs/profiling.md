# NVTX 性能分析标签

使用T-BLAEQ的NVTXProfiler适配实现，为Nsight Systems提供RAII主机范围。头文件为NVTXProfiler.cuh；来源、许可证与修改说明见third-party/NVTXProfiler/NOTICE。

## 构建开关

~~~bash
cmake -S . -B build-nvtx -DCMAKE_BUILD_TYPE=Release -DEMBERFIELD_ENABLE_NVTX=ON
cmake --build build-nvtx --parallel
~~~

EMBERFIELD_ENABLE_NVTX默认OFF。ON时CMake通过EmberField::emberfield的PUBLIC使用要求定义同名预处理宏；构建树及安装包的下游目标自动继承。关闭时不定义该宏，快捷宏展开为空，包括名称、颜色表达式均不求值，头文件不包含NVTX或其辅助标准库。修改开关后应重新构建所有使用相关头文件的目标，不在单个源文件中手动混用定义。

启用需要CUDA Toolkit的nvtx3/nvToolsExt.h。NVTX3为头文件接口；Unix平台由CMake传递动态加载支持库，不依赖已移除的旧libnvToolsExt。保持最低CMake 3.24要求，不依赖较新版本才提供的CUDA::nvtx3目标。

## 作用域接口

~~~cpp
#include <NVTXProfiler.cuh>

void assemble() {
    EMBERFIELD_NVTX_SCOPE("assembly");
    {
        EMBERFIELD_NVTX_SCOPE_COLOR("volume", emberfield::NvtxColor::Green);
        // 体积分与组装。
    }
    {
        EMBERFIELD_NVTX_SCOPE_RANDOM("boundary");
        // 边界积分。
    }
}
~~~

宏创建自动局部对象，退出当前C++作用域时弹出标签，异常路径同样释放。需要更短范围时使用花括号；不要把宏作为无花括号if/for的单条语句使用。局部变量名通过两层宏展开与__COUNTER__去重，同一行和同一作用域的多个标签均可使用。

启用时也可直接构造emberfield::NvtxProfiler并显式release()；重复release()无操作。isActive()表示对象是否尚未弹栈，与分析器是否附加无关。直接类型仅在开关开启时声明，跨配置代码使用快捷宏。

默认标签使用固定名称，无调用计数、字符串格式化、随机数初始化。固定色宏同样不启用编号；随机色宏会首次初始化线程局部随机源。确需编号可使用NvtxProfiler(name, mode, color, true)，计数按线程及完整名称保存，第二次开始追加#2等后缀；编号后的消息最多255字节，重复动态名称会增加线程局部映射存储，不建议用于细粒度热点。固定名称不截断。

NVTX push/pop属于当前线程的栈范围，必须同线程按后进先出顺序结束，因此包装类禁止复制和移动。宏内不做CUDA同步；范围覆盖的是主机提交区间，分析GPU执行应结合Nsight的CUDA流和内核时间线，不能直接将范围宽度解释为GPU完成耗时。

## 采集示例

~~~bash
nsys profile --trace=cuda,nvtx --sample=none -o case-profile \
  build-nvtx/emberfield example/cases/time-varying.json \
  --output-root /tmp --output emberfield-profile-result
~~~

结果目录须未被已有运行占用。采样和采集开销不包含在“关闭时空宏”的保证内。

## 范围层次与命名

正式程序及库按下表提供嵌套范围，example和test不插入标签；运行它们仍可采集其调用的正式库范围。
函数范围采用固定的C++限定名；模板类型参数不动态展开，常见线性代数重载以/Vector、/DenseMatrix、/CsrMatrix区分。
函数内部阶段使用/阶段名后缀。重复调用使用相同名称，通过时间线位置、嵌套深度与出现次数判断批次、迭代和层次，不生成动态编号字符串。

| 模块 | 颜色 | 主要范围 |
| --- | --- | --- |
| CLI | Rose | main、parse、硬件查询与启动信息 |
| Simulation | Yellow | runCase、advance、timeStep、初始组装、求解器重建、算例验证绑定、温度空间、准备、组装批次、积分表、时间参数采样、源/通量/对流/总载荷缓存更新、后处理、监控量与热平衡 |
| DataStructure | SkyBlue | 网格验证与索引构建、面拓扑、场快照验证、CSR验证、各种内存策略分配、释放、复制与转换 |
| FEM | Green | 有限元空间、几何准备、符号组装计划、坐标排序去重、局部槽排序、矩阵/向量批量累加 |
| Numerics | Cyan | 向量/稠密/稀疏运算、cuBLAS/cuSPARSE封装、迭代上下文创建释放与运算、线性系统构造、Dirichlet消元和温度投影 |
| Solver | Magenta | CG/BiCGStab setup/apply/solve/finish、每次迭代、真实残差刷新；Identity；Jacobi setup/apply及每次扫描；AMG setup/apply及每层构建、强连接图、两阶段匹配与吸收轮次、图收缩、Galerkin排序归并、前缀扫描、粗层分解与求解、V/W/F递归、前后平滑与限制延拓 |
| IO | Orange | 算例/资源限制读取解析与算例序列化、MatrixMarket、MSH加载保存及读入段、VTU数组编码、PVD、结果目录初始化、帧发布、文件写入、日志追加及摘要提交 |

颜色属于模块级提示，内部阶段默认沿用无显式颜色的标签。独立通用运算归其所属模块，不因被求解器调用而改名为求解器专属运算。

## 粒度与解释

- 不在CUDA内核、device函数、逐节点/逐积分点的短小数学操作、访问器、枚举转换、错误文本格式化中插入范围。CPU单元积分等细粒度函数由几何准备、积分表构建或批次调用者的范围覆盖；GPU活动在CUDA轨道查看。version等常量查询同样不打标。
- AMG递归范围按实际V/W/F分支命名，嵌套深度体现递归；W二次访问与F转V会形成独立子范围。setup/level反复出现对应逐层构建，matchingRound与absorptionRound体现有限轮次。
- 时变组装仅在确实进入更新分支时产生updateSource、updateFlux、updateConvection、updateLoad范围。缓存命中不产生虚假的更新范围；solverSetup同样仅在矩阵版本或步长要求重建时出现。
- 函数标签自进入函数体开始，不涵盖调用者实参求值和构造函数成员初始化列表。需要观察完整构造成本时查看外层阶段和内存/上下文子范围。
- 标签不添加同步、不改变流、不改变求解与I/O控制逻辑。GPU提交可能在主机范围退出后才执行，使用CUDA关联关系核对，不能累加父子范围作为总耗时。
- 启用后细粒度库调用和大量批次会产生较多事件；分析真实大算例时可缩短采集窗口。关闭开关可移除所有项目范围的执行开销。
