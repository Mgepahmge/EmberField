/** @file PreparationOptions.cuh
 * @brief 独立准备策略，避免条件与联合准备头文件循环依赖。
 */
#pragma once
#include <cstddef>
#include <limits>
namespace emberfield::problem {
    /** @brief 调用者控制的数值门槛与资源预算，不覆盖算例中的单位和厚度。
     * @code PreparationOptions o; o.maxCells_=10000; @endcode */
    struct PreparationOptions {
        double detTolerance_{128 * std::numeric_limits<double>::epsilon()}; ///< 归一化退化门槛。
        double maxCondition_{1e12}; ///< Frobenius条件数上限。
        double normalTolerance_{1e-10}; ///< 法向反向残差门槛。
        std::size_t maxCells_{10000000}; ///< 活动单元上限。
        std::size_t maxDofs_{10000000}; ///< 全DOF上限。
        std::size_t maxFacetUses_{40000000}; ///< 母侧访问上限。
        std::size_t maxAssociations_{100000000}; ///< 所有条件展开访问合计上限。
        std::size_t maxDiagnostics_{100}; ///< 保存诊断上限，截断仍保留失败。
    };
    /** @brief 校验准备策略。@param options 策略。@param dimension 维数。
     * @pre 无。@note 无返回值，非法策略抛invalid_argument。
     * @code validatePreparationOptions(o,2); @endcode */
    void validatePreparationOptions(const PreparationOptions& options, unsigned dimension);
} // namespace emberfield::problem
