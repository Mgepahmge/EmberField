/** @file Diagnostics.cuh
 * @brief 定义独立于解析库的算例诊断与资源限额。
 */
#pragma once
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

/** @brief 提供物理算例定义、校验与网格绑定，不承担文件编码和组装。 */
namespace emberfield::problem {
    /** @brief 区分阻断当前阶段的错误与不改变输入的警告。 */
    enum class Severity { Error, Warning };
    /** @brief 单条带稳定错误码的诊断，源码位置允许缺省。
     * @code Diagnostic d; d.code_="MATERIAL_GAP"; @endcode */
    struct Diagnostic {
        Severity severity_{Severity::Error}; ///< 严重级别。
        std::string code_{}; ///< 稳定机器可读错误码。
        std::string path_{}; ///< 逻辑字段路径，采用JSON Pointer形式。
        std::string objectId_{}; ///< 相关定义的id。
        std::optional<std::uint64_t> meshTag_{}; ///< 相关持久网格标签。
        std::string message_{}; ///< 中文解释。
        std::string suggestion_{}; ///< 可执行的修正建议。
        std::string source_{}; ///< 来源文件；程序化输入允许为空。
        std::optional<std::size_t> line_{}; ///< 一基行号，仅有依据时填写。
        std::optional<std::size_t> column_{}; ///< 一基列号。
    };
    /** @brief 控制读取和绑定的宿主资源消耗；不允许输入文件自行提高限额。
     * @code CaseLimits limits; limits.maxBytes_=1024*1024; @endcode */
    struct CaseLimits {
        std::size_t maxBytes_{16 * 1024 * 1024}; ///< 最大配置字节数。
        std::size_t maxValues_{1000000}; ///< 最大JSON值或程序化定义记录数。
        std::size_t maxStringBytes_{1024 * 1024}; ///< 单字符串字节上限。
        std::size_t maxNesting_{128}; ///< JSON嵌套上限。
        std::size_t maxSelectorDepth_{64}; ///< 选择器递归上限。
        std::size_t maxDiagnostics_{100}; ///< 保存诊断条数上限，至少为1。
    };
    /** @brief 限长诊断集合；截断不丢失错误状态。
     * @code Diagnostics d(10); d.add({Severity::Error,"INVALID","/time",{}, {},"非法时间"}); @endcode */
    class Diagnostics {
    public:
        /** @brief 创建集合。@param limit 最多保存条数。@pre limit大于0，否则抛invalid_argument。
         * @note 无返回值。@code Diagnostics d(100); @endcode */
        explicit Diagnostics(std::size_t limit = 100);
        /** @brief 添加诊断。@param value 独立拥有的诊断。@pre 无。
         * @note 无返回值，达到限额后仅更新状态和省略计数；分配失败抛标准异常。
         * @code d.add(diagnostic); @endcode */
        void add(Diagnostic value);
        /** @brief 合并另一集合。@param other 来源。@pre 允许自合并。
         * @note 无返回值，保留来源截断错误状态。@code d.append(other); @endcode */
        void append(const Diagnostics& other);
        /** @brief 查询是否出现错误；无参数无前提。@return 错误标志。
         * @code if(d.hasErrors()) return; @endcode */
        [[nodiscard]] bool hasErrors() const noexcept { return hasErrors_; }
        /** @brief 借用诊断数组；无参数无前提。@return 受当前对象生命周期限制的引用。
         * @code auto& entries=d.entries(); @endcode */
        [[nodiscard]] const std::vector<Diagnostic>& entries() const noexcept { return entries_; }
        /** @brief 查询省略条数；无参数无前提。@return 超出限额的条数。
         * @code auto n=d.omitted(); @endcode */
        [[nodiscard]] std::size_t omitted() const noexcept { return omitted_; }

    private:
        std::size_t limit_; ///< 保存上限。
        bool hasErrors_{}; ///< 包含被截断的错误状态。
        std::size_t omitted_{}; ///< 省略计数。
        std::vector<Diagnostic> entries_; ///< 拥有诊断文本。
    };
} // namespace emberfield::problem
