/** @file Validation.cuh
 * @brief 提供程序化定义与文件输入共用的静态校验。
 */
#pragma once
#include <Simulation/Case/Definition.cuh>

/** @brief 提供不读取网格、不创建GPU资源的定义校验。 */
namespace emberfield::problem {
    /** @brief 校验完整定义，不修改输入。
     * @param definition 程序化或文件构造的定义。@param limits 递归、记录、文本和诊断限额。
     * @pre 所有限额为正；无物理合法性前提。
     * @return 可聚合诊断；无错误仅表示静态定义合法，不表示网格绑定或可计算。
     * @throws std::invalid_argument 限额为零。标准分配异常向上传递。
     * @code auto report=validateCase(definition); if(report.hasErrors()) return; @endcode */
    [[nodiscard]] Diagnostics validateCase(const CaseDefinition& definition, CaseLimits limits = {});
    /** @brief 校验资源限额。@param limits 待检查限额。@pre 无。
     * @note 无返回值；任一零限额抛invalid_argument。@code validateLimits(limits); @endcode */
    void validateLimits(const CaseLimits& limits);
} // namespace emberfield::problem
