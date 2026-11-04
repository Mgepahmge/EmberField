# 文档目录

项目文档描述当前系统的设计、接口契约和使用方法。代码中的 Doxygen 注释提供
就近的 API 说明；本目录提供跨组件设计和用户操作说明。

| 文档 | 内容 |
| --- | --- |
| [用户手册](user-guide.md) | CLI、算例与资源配置、结果文件及顶层公共 API |
| [基础数据结构设计](data-structures.md) | 内存策略、资源所有权、复制与移动语义、矩阵存储及同步边界 |
| [线性代数接口](linear-algebra.md) | 运算、运算符、精度支持与执行约束 |
| [求解组件设计](solver.md) | 平滑器、预条件子、求解器及 AMG 循环的架构和契约 |
| [Matrix Market 用户手册](matrix-market.md) | 矩阵读取、命令行求解、参数、输出与退出码 |
| [网格与MSH](mesh.md) | 网格拓扑及文件读写 |
| [完整算例配置](case.md) | 物理参数、区域绑定和JSON契约 |
| [原生算例示例](native-case-example.md) | 稳态/瞬态配置、MSH分组及快速运行方法 |
| [外部算例转换](case-conversion.md) | Elmer和INP格式调研、转换工具、字段映射与使用边界 |
| [有限元空间](finite-element-space.md) | 形函数、自由度与局部映射 |
| [离散准备](thermal-preparation.md) | 几何、积分、条件与GPU局部热算子 |
| [全局组装与线性系统](assembly-system.md) | GPU稳定散加、稳态、后向Euler与对称消元 |
| [命令行与结果文件](results.md) | 算例运行、输出路径、VTU/PVD、监测、热量及故障语义 |

构建、安装和 Doxygen 生成方法见项目根目录的 [README](../README.md)。
文档只维护设计和使用所需的信息，不收录开发日志、修复经过、阶段进度或逐次测试记录。
