/** @file Case.cuh
 * @brief 提供严格JSON算例读取，不向公开接口泄露第三方类型。
 */
#pragma once
#include <Simulation/Case/Validation.cuh>
#include <iosfwd>
#include <map>

namespace emberfield::io {
    /** @brief 配置节点的来源位置，独立于物理模型。
     * @code SourceLocation s{"case.json",1,1}; @endcode */
    struct SourceLocation {
        std::string source_; ///< 文件名或调用者提供的来源标识。
        std::size_t line_{1}; ///< 一基行号。
        std::size_t column_{1}; ///< 一基列号。
    };
    /** @brief 加载结果；非法输入不返回部分定义。
     * @code if(result.definition_) use(*result.definition_); @endcode */
    struct CaseReadResult {
        std::optional<problem::CaseDefinition> definition_; ///< 完整合法定义。
        problem::Diagnostics diagnostics_; ///< 输入诊断。
        std::map<std::string, SourceLocation> locations_; ///< JSON Pointer到来源位置。
    };
    /** @brief 从流读取严格UTF-8 JSON，使用随项目分发的picojson。
     * @param input 借用输入流。@param baseDirectory 相对文件路径的绝对基目录。
     * @param source 来源名；允许为空。@param limits 加载限额。
     * @pre baseDirectory为绝对路径；限额为正。
     * @return 完整定义或诊断；重复键、未知字段、整数精度丢失和非标准JSON均拒绝。
     * @throws std::invalid_argument 非法调用前提；标准分配异常向上传递。
     * @code auto r=readCase(stream,std::filesystem::current_path(),"case.json"); @endcode */
    [[nodiscard]] CaseReadResult readCase(std::istream& input, const std::filesystem::path& baseDirectory,
                                          std::string source = {}, problem::CaseLimits limits = {});
    /** @brief 以配置文件所在目录为基准加载，不依赖运行目录解析内部路径。
     * @param path 配置路径。@param limits 限额。@pre 限额为正。
     * @return 定义或包含IO_ERROR的诊断；不加载网格、不创建输出目录。
     * @code auto r=loadCase("cases/plate.json"); @endcode */
    [[nodiscard]] CaseReadResult loadCase(const std::filesystem::path& path, problem::CaseLimits limits = {});
} // namespace emberfield::io
