/** @file Tui.cuh @brief CLI私有的终端探测及交互入口，库不依赖FTXUI。 */
#pragma once
#include <Simulation/Case/Run.cuh>
#include <string>
/** @brief EmberField终端观察与进度接口，不持有数值资源。 */
namespace emberfield::cli {
    /** @brief 本次终端可用性。@code TerminalCapabilities caps; @endcode */
    struct TerminalCapabilities {
        bool interactive_{}; ///< 输入输出均TTY且TERM可用。
        int columns_{}, rows_{}; ///< 当前窗口尺寸，未知为零。
    };
    /** @brief 解析后的呈现选择。 */
    enum class UiMode { Plain, Tui };
    /** @brief 查询终端，无参数无额外前提。@return TTY与当前尺寸，不初始化CUDA。
     * @code auto caps=terminalCapabilities(); @endcode */
    TerminalCapabilities terminalCapabilities() noexcept;
    /** @brief 选择呈现模式。@param requested auto/tui/plain。@param capabilities 当前终端。
     * @pre 无。@return 实际模式；显式不可用tui抛runtime_error。
     * @code auto mode=selectUiMode("auto",terminalCapabilities()); @endcode */
    UiMode selectUiMode(const std::string& requested, TerminalCapabilities capabilities);
    /** @brief 主线程展示并等待唯一计算线程，始终join后返回。
     * @param input 算例。@param output 结果目录。@param limits 资源限制。
     * @param noWait 完成后立即退出。@pre TUI构建且终端可用。
     * @return runCase退出码，异常在恢复终端并join后传播。
     * @code auto code=runTui(input,output,limits,true); @endcode */
    int runTui(const std::filesystem::path& input, const std::filesystem::path& output,
               const problem::ResourceLimits& limits, bool noWait);
} // namespace emberfield::cli
