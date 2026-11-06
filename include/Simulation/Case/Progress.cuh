/** @file Progress.cuh @brief 数值组装批边界进度，不改变CUDA执行顺序。 */
#pragma once
#include <cstddef>
/** @brief EmberField终端观察与进度接口，不持有数值资源。 */
namespace emberfield::problem {
    /** @brief 每条计数是一个独立工作段，不混用不同循环的分母。 */
    enum class ProgressWork { Cells, Boundaries, VolumeLoad, Flux, Convection };
    /** @brief 已完成批的有界计数。@code ProgressInfo p; @endcode */
    struct ProgressInfo {
        ProgressWork work_{ProgressWork::Cells}; ///< 当前工作类别。
        std::size_t completed_{}; ///< 已完成实体数。
        std::size_t total_{}; ///< 本段实体总数；零表示空段。
        bool determinate_{true}; ///< 是否已知分母。
    };
    /** @brief 同步观察批边界，借用期内须有效。@code ProgressObserver observer; @endcode */
    class ProgressObserver {
    public:
        virtual ~ProgressObserver() = default; ///< 多态释放。
        /** @brief 接收已有计数。@param info 本段快照。@pre 不抛异常或保留引用。
         * @note 无返回值。@code observer.onProgress(info); @endcode */
        virtual void onProgress(const ProgressInfo& info) noexcept { (void)info; }
    };
} // namespace emberfield::problem
