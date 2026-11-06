/** @file ConvergenceProgress.cuh @brief 用首轮残差到真实阈值的距离展示收敛。 */
#pragma once
#include <Solver/Iterative/IterationObserver.cuh>
#include <algorithm>
#include <cmath>
#include <optional>
/** @brief EmberField终端观察与进度接口，不持有数值资源。 */
namespace emberfield::cli {
    /** @brief 单次apply的标量展示状态，不参与判停。
     * @code ConvergenceProgress progress; progress.update(info); @endcode */
    class ConvergenceProgress {
    public:
        /** @brief 清空本次状态，无参数无前提无返回。@code progress.reset(); @endcode */
        void reset() noexcept { *this = ConvergenceProgress{}; }
        /** @brief 更新残差展示。@param info 已有求解标量。@pre 无。
         * @note 无返回值；首轮只接受Iteration且编号为1；Final不补造基线。
         * @code progress.update(info); @endcode */
        void update(const IterationInfo<double>& info) noexcept {
            if (info.kind_ == IterationKind::Initial) {
                reset();
            }
            iteration_ = info.iteration_;
            residual_ = info.residualNorm_;
            tolerance_ = info.tolerance_;
            residualKind_ = info.residualKind_;
            status_ = info.status_;
            fraction_.reset();
            if (!std::isfinite(*residual_) || !std::isfinite(*tolerance_) || *residual_ < 0 || *tolerance_ < 0) {
                return;
            }
            if (info.kind_ == IterationKind::Iteration && info.iteration_ == 1 && !firstResidual_) {
                firstResidual_ = *residual_;
            }
            if (*residual_ <= *tolerance_) {
                fraction_ = 1;
                return;
            }
            if (firstResidual_ && *firstResidual_ > *tolerance_) {
                // 先归一化，避免r0-tol的差与极大r的商发生中间溢出。
                const auto denominator = 1 - *tolerance_ / *firstResidual_;
                fraction_ = std::clamp(1 - (*residual_ / *firstResidual_ - *tolerance_ / *firstResidual_) / denominator,
                                       0.0, 1.0);
            }
        }
        /** @brief 数值验收通过，无参数无前提无返回。@code progress.accept(); @endcode */
        void accept() noexcept { accepted_ = true; }
        std::optional<double> fraction() const noexcept { return fraction_; } ///< 查询比例，无参数无前提。
        std::optional<double> residual() const noexcept { return residual_; } ///< 查询残差，无参数无前提。
        std::optional<double> tolerance() const noexcept { return tolerance_; } ///< 查询阈值，无参数无前提。
        int iteration() const noexcept { return iteration_; } ///< 查询轮号，无参数无前提。
        ResidualKind residualKind() const noexcept { return residualKind_; } ///< 查询来源，无参数无前提。
        std::optional<SolveStatus> status() const noexcept { return status_; } ///< 查询终止报告，无参数无前提。
        bool accepted() const noexcept { return accepted_; } ///< 查询验收，无参数无前提。
    private:
        std::optional<double> firstResidual_, residual_, tolerance_, fraction_; ///< 首轮、当前、阈值与比例。
        int iteration_{}; ///< 当前轮号。
        ResidualKind residualKind_{ResidualKind::True}; ///< 当前来源。
        std::optional<SolveStatus> status_; ///< Final状态。
        bool accepted_{}; ///< 投影后数值验收通过。
    };
} // namespace emberfield::cli
