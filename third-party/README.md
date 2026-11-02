# 第三方组件

本目录随源代码分发，不参与项目代码格式化。EmberField的MIT许可证不替代各组件原始许可证。

| 组件 | 用途 | 来源与版本边界 | 许可证 |
| --- | --- | --- | --- |
| picojson | IO层JSON解析 | 按项目所有者授权原样复制自服务器T-BLAEQ的src/picojson.h；副本未声明精确发布版本，不猜测版本号；上游 https://github.com/kazuho/picojson | BSD-2-Clause，原文见picojson/LICENSE，头文件保留原始声明 |
| CLI11 | 主程序命令行解析 | 原样复制自T-BLAEQ；头文件版本2.6.1；上游 https://github.com/CLIUtils/CLI11 | BSD-3-Clause，原文见CLI11/LICENSE |

本项目未修改picojson原文。IO适配器启用PICOJSON_USE_INT64，保留库默认的区域设置适配，JSON输入始终使用点号小数；使用独立解析上下文拒绝重复键、限制递归与值数量。数字词法预检拒绝超出int64范围的整数，防止picojson回退double损失精度。适配代码位于src/IO/Case.cu，不写入第三方头文件。

复制依据与哈希：

- 原头文件：`/home/mingyu/CLionProjects/T-BLAEQ/src/picojson.h`。
- 原许可证：`/home/mingyu/CLionProjects/T-BLAEQ/THIRD_PARTY_LICENSES/picojsonLICENSE.`。
- picojson.h SHA256：`5ddf7276d04926da7be243e7af49258e78cc27278ee9097ba45b942c7a6b5f9d`。
- LICENSE SHA256：`5585fe141cc7bb08c953f3859db608852968d0bbc625b9b6d95c0bd6349bacb6`。

安装产物包含原始头文件与本说明（share/EmberField/third-party），以及许可证（share/licenses/EmberField/picojson）。公开EmberField头文件不暴露picojson，不需要下游安装JSON包或在公开include路径中加入本组件。

CLI11来源为`/home/mingyu/CLionProjects/T-BLAEQ/src/CLI11.hpp`和`THIRD_PARTY_LICENSES/CLI11LICENSE`，按所有者授权复制，未修改原文。头文件内嵌通知和许可证文件的年份保持各自原样。

- CLI11.hpp SHA256：`8bad421104bbac1d763be60e3f833768e72c2d9f6fa14a6d19162979ba97c2e9`。
- LICENSE SHA256：`79befe5e6cfba74faaf147acb32c645e65cb03664c6a7c7ddb9fef2ec937ed9f`。

CLI11仅作为主程序私有头文件依赖，不暴露于库的公共接口。安装保留原件（share/EmberField/third-party/CLI11）及许可证（share/licenses/EmberField/CLI11），不需要网络下载或单独安装。
