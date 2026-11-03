/** @file ResourceLimits.cuh @brief 严格读取独立于算例物理配置的资源预算JSON。 */
#pragma once
#include <Simulation/Case/ResourceLimits.cuh>
#include <filesystem>
#include <iosfwd>

/** @brief 资源文件适配层，不向计算模块暴露JSON类型。 */
namespace emberfield::io {
    /** @brief 从流事务读取与文件接口相同的严格资源协议。
     * @param input 借用流。@param source 诊断来源标识，可为空。
     * @pre 流可读。@return 完整策略；失败不发布部分结果。
     * @throws std::runtime_error 语法、协议、流读取或固定解析预算错误；分配异常传播。
     * @code auto limits=readResourceLimits(stream,"limits.json"); @endcode */
    [[nodiscard]] problem::ResourceLimits readResourceLimits(std::istream& input, std::string source = {});
    /** @brief 事务读取资源文件，省略的预算保留模块默认值。
     * @param path UTF-8 JSON文件路径；相对路径按调用工作目录解释。
     * @pre 文件可读；schemaVersion为1，计数字段是可表示的正整数字面量。
     * @return 独立组合策略；失败不返回部分结果，不访问CUDA或输出目录。
     * @throws std::runtime_error 文件错误、语法、重复/未知键、类型或预算错误，包含来源和字段路径。
     * @note 文件自身按固定16MiB、1000000值、128层和1MiB字符串解析，不能自行扩大解析预算。
     * 标准分配异常向上传递。整数不经过double转换，最大支持INT64_MAX且须适合目标类型。
     * @code auto limits=loadResourceLimits("limits.json"); @endcode */
    [[nodiscard]] problem::ResourceLimits loadResourceLimits(const std::filesystem::path& path);
} // namespace emberfield::io
