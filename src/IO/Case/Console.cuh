/** @file Console.cuh @brief 普通终端消息接收端，无ANSI控制序列。 */
#pragma once
#include <Simulation/Case/RunObserver.cuh>
/** @brief EmberField终端观察与进度接口，不持有数值资源。 */
namespace emberfield::cli {
    /** @brief 保持普通模式逐行输出，不逐次打印迭代。@code ConsolePresenter presenter; @endcode */
    class ConsolePresenter final : public problem::RunObserver {
    public:
        /** @brief 输出文本消息，捕获流异常。@param event 临时事件。
         * @pre 串行调用。@note 无返回值。@code presenter.onEvent(event); @endcode */
        void onEvent(const problem::RunEvent& event) noexcept override;
    };
} // namespace emberfield::cli
