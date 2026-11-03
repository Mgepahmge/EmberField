# 运行资源限制

```bash
emberfield case.json --resource-limits limits.json
```

根目录的 `limits.json` 是随项目版本管理的完整示例，填写模块默认值。程序不会自动搜索或加载它。
安装时该示例同时放在安装前缀的 `share/EmberField/limits.json` 中（随平台的数据目录约定调整）。
相对路径按启动工作目录解析；资源文件只管理执行预算，物理模型和求解设置仍来自算例文件。
不指定参数时保留原行为。文件只支持 `schemaVersion: 1`，版本字段必填，其他分组和字段均可省略。

## 合并与验证

- `case`、`mesh`、`space`、`preparation`、`assembly`、`integration` 中省略的字段保留模块默认值。
- `output` 逐字段覆盖算例 `output.limits`；未指定的字段保留算例值，算例未指定时使用模块默认值。
- 输出覆盖在算例语义验证、绑定和后处理计划建立之前生效；有效值会记录在结果摘要的 `input.effectiveCase.output.limits`。
- 所有预算必须为正整数字面量，不接受零、负数、布尔、字符串、浮点或指数写法。整数最大为 `INT64_MAX`，同时不得超出目标类型范围。
- 拒绝未知字段、重复键、错误类型、尾随内容、非法UTF-8及不支持的版本；错误包含文件来源和JSON字段路径，读取失败不会创建结果目录。
- 资源文件本身始终使用固定解析限额：16MiB、1000000个JSON值、128层嵌套、单字符串1MiB。文件内容不能提高这些解析限额。

以下示例仅扩大绑定记录及后处理缓存，其余保留原值：

```json
{
  "schemaVersion": 1,
  "case": {"maxValues": 5000000},
  "output": {"maxHostBytes": 4294967296, "maxDeviceBytes": 2147483648}
}
```

## 字段

下表是模块原生默认值；`output` 未覆盖时优先保留算例中的值。字节字段以字节为单位，其余为计数或深度。

| 分组 | 字段 | 默认值 | 含义 |
| --- | --- | --- | --- |
| case | maxBytes | 16777216 | 算例JSON输入字节 |
| case | maxValues | 1000000 | JSON值、物理组索引、区域选择访问和绑定记录的共用计数上限，具体计数独立检查 |
| case | maxStringBytes | 1048576 | 单个字符串或键的字节数 |
| case | maxNesting | 128 | 算例JSON嵌套深度 |
| case | maxSelectorDepth | 64 | 区域选择器递归深度 |
| case | maxDiagnostics | 100 | 保存诊断条数；截断不丢失错误状态 |
| mesh | maxRecords | 100000000 | MSH单数组记录数 |
| mesh | maxBytes | 2147483648 | 整个MSH输入文件字节 |
| space | maxCells | 1000000 | 活动单元数 |
| space | maxLocalDofs | 4000000 | 局部自由度总数 |
| space | maxGlobalDofs | 4000000 | 全局自由度数 |
| space | maxDiagnostics | 100 | 空间构建保存诊断条数 |
| preparation | maxCells | 10000000 | 准备阶段活动单元数 |
| preparation | maxDofs | 10000000 | 准备阶段自由度数 |
| preparation | maxFacetUses | 40000000 | 母侧访问数 |
| preparation | maxAssociations | 100000000 | 条件展开关联总数 |
| preparation | maxDiagnostics | 100 | 准备阶段保存诊断条数 |
| assembly | maxDofs | 10000000 | 符号组装全局维度 |
| assembly | maxStencils | 10000000 | 局部编号模板数 |
| assembly | maxLocalEntries | 64000000 | 所有局部方阵条目总数 |
| assembly | maxNonzeros | 64000000 | 规范CSR条目数，包含显式零 |
| integration | maxElements | 65536 | 单GPU积分批实体数 |
| integration | maxScalars | 16777216 | 单GPU积分批中每个平坦数组的标量数 |
| output | maxPoints | 4000000 | 活动输出顶点数 |
| output | maxCells | 1000000 | 活动输出单元数 |
| output | maxHostBytes | 1073741824 | 后处理主机缓存预算 |
| output | maxDeviceBytes | 536870912 | 后处理设备缓存预算 |
| output | maxFrames | 1000000 | 输出帧数 |
| output | maxMonitors | 1000 | 监测定义数 |
| output | maxOutputBytes | 107374182400 | 输出文件总字节 |

## 资源语义

这些是各组件的独立拒绝阈值。后处理内存检查包含保守估计，不等同于实际分配量；
它们不能保证整个程序的主机内存、显存或磁盘使用量。求解器和第三方库仍可能申请额外工作空间。
设置较大预算不会预分配对应内存，也不能避免分配失败。调用者负责根据机器和任务安排预算。

GPU积分的实际批大小同时受内部批大小和积分预算限制；提高预算不会自动放大批大小。
`batchElements`、`batchCells`、物理/数值容差、迭代次数和时间步不属于本资源文件协议。
索引可表示性、溢出及输入合法性检查始终保留。
