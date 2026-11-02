# 有限元空间与自由度

当前有限元模块可以从只读Mesh、活动单元及有限元键构建未施加物理约束的标量空间。它独立于JSON、MSH文件、材料和CUDA存储；算例适配器负责将已绑定算例的离散意图转交本模块。

## 支持范围

- 完整一阶Triangle3的连续Lagrange P1：每顶点一个温度自由度，参考形函数为`1-xi-eta, xi, eta`。
- 完整一阶Tet4的连续Lagrange P1：参考形函数为`1-xi-eta-zeta, xi, eta, zeta`。
- 每个空间只包含一种活动维数。高阶、自定义布局及其他单元明确报不支持，不截取角点降阶。

几何阶次与温度阶次分别管理。当前两者均要求为一阶；不代表这两个概念相同。基函数求值返回参考梯度，不包含Jacobian、积分权重或物理坐标缩放。允许有限参考域外坐标进行显式外插，不裁剪。

## 独立使用

```cpp
#include <FEM/Space/Space.cuh>

using namespace emberfield;
std::vector<mesh::CellRef> active{{0, 0}, {0, 1}};
std::vector<fem::ElementKey> keys(active.size(), {mesh::CellFamily::Triangle});
auto result = fem::buildFiniteElementSpace(mesh, active, keys);
if (result.space_) {
    auto localToGlobal = result.space_->cellDofs(0);
    auto side = fem::facetDofs(*result.space_, {{0, 0}, 2});
}
```

`mesh`应为共享只读网格所有权。调用者必须保证网格共形、无悬挂节点；构建器不会从坐标证明全域共形、无重叠或Jacobian有效。不得通过其他可变别名修改、移动或替换底层网格。空间独立拥有映射和有限元键，发布后可并发只读查询。

## 编号与边界

节点tag、Mesh节点索引、单元局部DOF、空间全局DOF是不同身份。当前全局DOF按活动顶点的Mesh索引升序确定，保留每个单元的原始局部顺序；不依赖最大tag分配内存。活动单元顺序变化不改变同一Mesh中的节点编号；Mesh节点存储顺序变化可以改变编号，跨文件关联使用持久tag。

同节点身份跨单元共享DOF，包括材料界面；同坐标但不同身份不自动合并。仅活动单元的顶点参与编号，未使用节点与独立边界单元不增加未知数。侧DOF按母单元的局部参考侧顺序返回，不使用用于匹配的排序节点表，不承诺实际外法向。

保留所有活动DOF，包括未来Dirichlet节点；不执行消元、初温分配、共享条件冲突检查或纯Neumann可解性判定。周期、分区与ghost元数据在通用入口中产生未施加关系的警告，不建立周期或分布式编号。算例适配器更严格：包含周期信息的网格在尚未定义周期物理策略时被拒绝。

## 从算例准备空间

```cpp
#include <Simulation/Case/TemperatureSpace.cuh>
auto space = emberfield::problem::buildTemperatureSpace(*binding.bound_);
```

绑定必须已经成功；本接口不要求整个算例已可执行，也不更改原始绑定报告。编号成功只表明空间建立，几何、积分、共享DOF上的物理条件、组装与结果输出仍待后续组件完成。

```bash
./build/example/space-inspect example/cases/steady.json
./build/example/space-inspect example/cases/transient.json
```

可选第二个参数是MSH往返输出文件，会按原文件版本写入该路径；避免使用需要保留的已有文件。仍须满足现有MSH写出器的可表示性限制，例如2.2输出要求标签不超过int32；随附plate.msh故意含大标签用于输入验证，不能用2.2写出。程序核对活动顶点与映射，并在可写出的网格往返后按单元tag与局部DOF对应节点tag比较。退出0表示空间检查通过，不表示热传导求解完成。

## 错误与资源

`SpaceLimits`默认上限为100万活动单元、400万局部DOF、400万全局DOF及100条诊断，可由调用者显式调整。输入不能自行提高上限。索引累加和转换仍独立检查溢出，不能因限额较大就省略。

空活动集、错引用、重复活动单元、单元内重复节点、重复完整连接、不支持布局、维数不一致和超限通过稳定错误码返回；错误时空间为空，诊断截断也保留错误状态。非法零限额抛`invalid_argument`，系统分配异常向上传播。查询越界抛`out_of_range`。`DofIndex`为64位，未来转为GPU矩阵int索引必须另行检查，不允许静默截断。
