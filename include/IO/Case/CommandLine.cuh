/** @file CommandLine.cuh @brief 与参数解析器无关的CLI路径规则。 */
#pragma once
#include <filesystem>
#include <optional>
namespace emberfield::io {
    /** @brief 查询当前实际可执行文件目录，无参数。@pre 操作系统提供进程路径。
     * @return 绝对目录；无法查询时runtime_error。@code auto root=executableDirectory(); @endcode */
    std::filesystem::path executableDirectory();
    /** @brief 构造单一安全输出目录。@param input 配置路径。@param root 根目录。
     * @param name 可选单级目录名，缺省为input.stem()。@pre root非空。
     * @return 绝对规范化root/name；非法名称invalid_argument。
     * @code auto p=outputDirectory("case.json",root,{}); @endcode */
    std::filesystem::path outputDirectory(const std::filesystem::path& input, const std::filesystem::path& root,
                                          const std::optional<std::string>& name = {});
} // namespace emberfield::io
