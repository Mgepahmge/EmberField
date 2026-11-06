/** @file Console.cu @brief 普通文本呈现实现。 */
#include <iostream>
#include "Console.cuh"
namespace emberfield::cli {
    void ConsolePresenter::onEvent(const problem::RunEvent& event) noexcept {
        if (event.kind_ != problem::RunEventKind::Message) {
            return;
        }
        try {
            (event.error_ ? std::cerr : std::cout) << event.message_ << std::flush;
        }
        catch (...) {
        }
    }
} // namespace emberfield::cli
