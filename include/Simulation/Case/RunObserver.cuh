/** @file RunObserver.cuh @brief 解耦运行编排与终端呈现的同步事件协议。 */
#pragma once
#include <Simulation/Case/Progress.cuh>
#include <Solver/Iterative/IterationObserver.cuh>
#include <atomic>
#include <optional>
#include <string_view>
/** @brief EmberField终端观察与进度接口，不持有数值资源。 */
namespace emberfield::problem {
    /** @brief 原子取消请求，仅在运行编排安全边界消费。
     * @code RunControl control; control.requestCancel(); @endcode */
    class RunControl {
    public:
        /** @brief 请求取消，无参数无前提。@note 无返回，不中断CUDA操作。
         * @code control.requestCancel(); @endcode */
        void requestCancel() noexcept { cancelled_.store(true, std::memory_order_relaxed); }
        /** @brief 查询请求，无参数无前提。@return 是否取消。@code bool c=control.cancelled(); @endcode */
        bool cancelled() const noexcept { return cancelled_.load(std::memory_order_relaxed); }

    private:
        std::atomic<bool> cancelled_{false}; ///< 可跨线程请求；不与信号处理器共享。
    };
    /** @brief 当前运行阶段。 */
    enum class RunStage {
        Loading,
        Preparing,
        Assembling,
        BuildingSystem,
        SettingUp,
        Solving,
        Checking,
        Postprocessing,
        Saving,
        Finalizing,
        Completed,
        Failed,
        Cancelled
    };
    /** @brief 事件含义，StepCommitted仅表示已发布且已接受的实际时间步。 */
    enum class RunEventKind { Stage, Metadata, StepStarted, SolveAccepted, StepCommitted, Message, Finished };
    /** @brief 展示所需标量及借用字符串，回调返回后不可保留引用。
     * @code CaseSummary summary; @endcode */
    struct CaseSummary {
        std::string_view name_, mesh_, analysis_, solver_; ///< 名称、路径、分析及方法。
        std::size_t nodes_{}, cells_{}, dofs_{}, nonZeros_{}; ///< 网格及矩阵规模。
        double relativeTolerance_{}, absoluteTolerance_{}; ///< 求解配置。
    };
    /** @brief 同步事件视图，不拥有数值资源。@code RunEvent event; @endcode */
    struct RunEvent {
        RunEventKind kind_{RunEventKind::Stage}; ///< 当前事件类型。
        RunStage stage_{RunStage::Loading}; ///< 当前阶段。
        std::string_view message_; ///< 临时文本，呈现端立即复制且清理控制字符。
        bool error_{}; ///< 消息是否发往标准错误；仅Message有效。
        std::optional<CaseSummary> summary_; ///< 仅Metadata有值。
        std::size_t step_{}, totalSteps_{}; ///< 实际步及预演总步数；初态为零。
        std::optional<double> time_, dt_, acceptedResidual_; ///< 当前时间及接受残差。
        double elapsedS_{}; ///< 运行墙钟秒数。
        int exitCode_{}; ///< 仅Finished有效，退出码与runCase一致。
    };
    /** @brief 可选运行观察者，所有回调同步、无异常且不可重入。
     * @code struct Observer : RunObserver {}; @endcode */
    class RunObserver : public IterationObserver<double>, public ProgressObserver {
    public:
        /** @brief 接收运行事件。@param event 临时视图。@pre 不保留引用或抛异常。
         * @note 无返回值。@code observer.onEvent(event); @endcode */
        virtual void onEvent(const RunEvent& event) noexcept { (void)event; }
    };
} // namespace emberfield::problem
