# 原生算例示例

项目提供一组可直接运行的 EmberField 原生算例：`example/cases/steady.json`（稳态）、
`transient.json`（瞬态）和共享网格`plate.msh`。
此算例用于演示文件结构及字段之间的关系，不代表特定材料的实验结果。

## 运行

在项目根目录构建后运行：

```bash
build-output/emberfield example/cases/steady.json --output-root results --output plate-steady
build-output/emberfield example/cases/transient.json --output-root results --output plate-transient
```

Windows可执行文件位于`build-output/`对应的配置目录中。算例配置与网格的相对路径
相对于JSON文件所在目录解析。程序会在`results/plate-steady`和
`results/plate-transient`中生成VTU结果及运行摘要；请为每次运行选择新的输出目录。

只运行首步快速检查时，可以复制一份JSON，将`time.maxSteps`改为所需上限；
稳态示例无需修改时间字段。也可以只读查看网格分组和配置：

```bash
build-output/example/case-inspect example/cases/steady.json
```

可执行文件的具体生成路径可能随CMake生成器不同而变化。

## 文件之间的关系

`plate.msh`使用MSH 2.2 ASCII，包含4个节点、2个三角形体单元及4条边界线。
坐标以毫米数值写入；两个JSON用`coordinateScale: 0.001`转换为米。
节点/单元的持久标签不要求连续。物理组为：二维tag 11=`body`；一维tag 21=`bottom`、
22=`right`、23=`top`、24=`left`。组的dimension必须与其所选实体维数一致。

两份JSON均展示：网格文件、分析类型与区域、材料、材料区域分配、体积热源、
规定温度/对流/外向热流边界、未指定边界默认绝热、P1离散及温度VTU输出。
稳态只需导热率；瞬态还需要密度和比热、覆盖活动域的初始温度以及时间范围、步长、
积分方法和最大步数。输出配置可以指定目录，也可以由CLI的`--output-root`和`--output`
统一覆盖结果位置。

输入单位约定为：几何缩放后SI，绝对温度K，导热率W/(m K)，密度kg/m³，
比热J/(kg K)，体源W/m³，外向热流W/m²，换热系数W/(m² K)。
`plate.msh`是人为构造的小型演示网格，不是工程精度或性能基准。

完整schema、区域选择、边界组合及错误语义见[算例输入与网格绑定](case.md)。
