# 网格模型与MSH文件 {#mesh_io}

## 边界与依赖

`Mesh/Mesh.cuh` 定义宿主侧网格拓扑模型；`IO/Msh.cuh` 定义独立文件适配器。
网格可以直接由程序构造，不需要MSH文件、CUDA分配、材料参数、边界条件或求解器。
本模块不进行FEM组装、自由度编号、Jacobian有效性判断或温度场求解。

节点持久标签和连续数组索引是不同概念：`Node::tag_` 保留身份，`ElementBlock::nodes_`
使用节点数组索引。局部连接次序不能排序；高阶节点的次序属于 `ElementType::referenceLayout_`
的契约，内置complete、edge-only、mini等布局沿用Gmsh参考单元次序。

`MeshData` 是可编辑构建值；`Mesh` 组合拥有经过验证的数据与标签索引，只提供只读访问。
修改时复制 `data()`、编辑并重新构造。复制独立持有所有数据；移动后来源为空网格；
复制赋值采用复制交换，分配失败时原对象保持有效。不同实例可独立使用，同一实例修改需要外部同步。

支持混合维度/单元族、高阶单元、变长多边形/多面体连接、显式有向单元边界、
实体边界与嵌入关联、物理分组、参数坐标、周期映射、分区和幽灵信息。
允许局部网格引用文件外的分类实体；不凭缺失关联推断物理边界，也不要求网格是封闭流形。

## 读取与保存

```cpp
#include <IO/Msh.cuh>

auto document = thermofem::io::loadMsh("input.msh");
const auto& mesh = document.mesh_;
auto firstCell = mesh.connectivity(0, 0);
thermofem::io::saveMsh("output.msh", document,
                     thermofem::io::MshEncoding::Binary);
```

流接口 `readMsh` / `writeMsh` 可用于内存、网络或自定义流；文件接口按二进制方式打开，
避免平台换行转换破坏二进制内容。输出ASCII使用经典locale与 `max_digits10` 精度。

支持MSH4.1和MSH2.2的ASCII/二进制读取与保存。4.1输出使用64位size_t字段；
输入支持4/8字节size_t和相反字节序。2.2按标准使用32位标签和8字节double。
重复Nodes/Elements段可拼接，重复实体/物理组声明必须一致；节点可以后于单元出现。

4.1解释并保留PhysicalNames、Entities、PartitionedEntities、Nodes、Elements、Periodic、
GhostElements、Parametrizations。2.2的逐单元物理标签、分类及分区成员关系进入网格模型，
额外原始标签保存在IO文档中。`MshDocument` 组合网格与这些兼容性信息；版本、扩展段等不进入Mesh。

未知ASCII段可作为文件扩展保留，但不参与拓扑解释；未知二进制段被拒绝，因为无类型信息无法
安全定位段末。包含未知扩展的文档只能输出ASCII。NodeData、ElementData、ElementNodeData和
InterpolationScheme明确不属于网格输入，不会静默丢弃。MSH4.0/4.2等版本不在本适配器范围。

单元编号目录支持1至140中已登记的固定连接布局，包括零阶、完整高阶、不完整高阶、
mini和特殊边界类型；34、35、69、76至78等无标准固定连接布局的编号明确拒绝。
核心模型仍可表示自定义、变长及多面体拓扑；“可表示网格”不等于“任意格式均能保存该网格”。

4.1保存要求节点和单元具有明确分类，不自动猜测。无法表达的独立单元边界、嵌入关系、
逐单元分组覆盖和逐单元分区须由调用者显式转换；2.2不能表达一般实体图和参数节点。
2.2保存会在内存中重读并核对网格值，拒绝改变结构或元数据的降级。默认保存版本为4.1，
读取2.2后如需原样保存应显式指定 `MshVersion::V22`。

```cpp
thermofem::io::saveMsh("legacy-copy.msh", document,
    thermofem::io::MshEncoding::Ascii, thermofem::io::MshVersion::V22);
```

## 错误和资源限制

`Mesh` 构造对重复/零节点或单元标签、非法连接偏移、越界索引、非有限数值、参数维度及
周期/分区关系等抛出 `std::invalid_argument`；查询不存在的标签或单元抛出 `std::out_of_range`。
验证不证明单元几何质量、物理单位一致性或高阶几何非退化。

IO格式错误、不支持版本/布局、资源限额、打开和流错误抛出 `std::runtime_error`；
标准分配异常照常传播。`MshLimits` 默认每数组最多一亿记录、总输入2GiB，可由调用者调整。
编解码在宿主内存进行，输入和输出各可能有完整缓冲副本，不承诺流式常量空间。
保存会在打开目标文件之前完成可表示性检查和编码；磁盘写入失败仍可能留下不完整文件，
不提供文件系统事务替换保证。

## 示例与验证

```sh
./build/example/mesh-roundtrip input.msh output.msh binary 4.1
./build/example/mesh-roundtrip legacy.msh legacy-copy.msh ascii 2.2
./build/test/thermofem_tests --gtest_filter='Mesh.*:Msh.*'
```

可选交叉验证需要Python可导入独立Gmsh，建议4.14.1；不属于库依赖，也不纳入默认CI：

```sh
python3 scripts/verify_mesh_gmsh.py build/example/mesh-roundtrip
```

其通过Gmsh生成二阶二维/三维、周期及分区网格，交由本库重编码，再由Gmsh重读，比较
持久标签、坐标、局部连接次序、分类、物理组及分区。自洽往返测试不能替代这种独立检查，
这些算例也不等于对所有单元布局、所有畸形输入的证明。

格式依据：[Gmsh 4.14.1参考手册源文档](https://github.com/live-clones/gmsh/blob/gmsh_4_14_1/doc/texinfo/gmsh.texi)，
具体字段核对同版MSH4及离散曲面参数化实现。本库不链接Gmsh运行库。
