/** @file Tui.cu @brief FTXUI呈现、固定频率快照与安全计算线程生命周期。 */
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <deque>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <poll.h>
#include <sstream>
#include <stdexcept>
#include <sys/ioctl.h>
#include <thread>
#include <unistd.h>
#include "Console.cuh"
#include "ConvergenceProgress.cuh"
#include "Tui.cuh"
#ifdef EMBERFIELD_BUILD_TUI
#include <ftxui/component/component.hpp>
#include <ftxui/component/event.hpp>
#include <ftxui/component/loop.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>
#endif
/** @brief EmberField终端观察与进度接口，不持有数值资源。 */
namespace emberfield::cli {
    TerminalCapabilities terminalCapabilities() noexcept {
        TerminalCapabilities caps;
        const char* term = std::getenv("TERM");
        caps.interactive_ =
            isatty(STDIN_FILENO) && isatty(STDOUT_FILENO) && term && *term && std::string_view(term) != "dumb";
        winsize size{};
        if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &size) == 0) {
            caps.columns_ = size.ws_col;
            caps.rows_ = size.ws_row;
        }
        return caps;
    }
    UiMode selectUiMode(const std::string& requested, TerminalCapabilities caps) {
        if (requested != "auto" && requested != "tui" && requested != "plain") {
            throw std::invalid_argument("未知界面模式");
        }
        bool available = caps.interactive_ && caps.columns_ >= 40 && caps.rows_ >= 18;
#ifndef EMBERFIELD_BUILD_TUI
        available = false;
#endif
        if (requested == "tui" && !available) {
            throw std::runtime_error("TUI不可用：需要启用TUI构建，交互终端至少40列18行");
        }
        return requested != "plain" && available ? UiMode::Tui : UiMode::Plain;
    }
#ifdef EMBERFIELD_BUILD_TUI
    namespace {
        using Clock = std::chrono::steady_clock;
        volatile std::sig_atomic_t terminalSignal = 0; ///< 只在信号处理器写，主线程读取。
        /** @brief 信号处理仅设标志。@param signal 信号号。@pre 无。@note 无返回值。
         * @code receiveSignal(SIGINT); @endcode */
        void receiveSignal(int signal) { terminalSignal = signal; }
        /** @brief 覆盖FTXUI处理器，退出本作用域时先恢复给FTXUI。
         * @code TerminalSignalGuard guard; @endcode */
        class TerminalSignalGuard {
        public:
            /** @brief 注册信号，无参数无前提无返回。@code TerminalSignalGuard g; @endcode */
            TerminalSignalGuard() :
                interrupt_(std::signal(SIGINT, receiveSignal)), termination_(std::signal(SIGTERM, receiveSignal)),
                hangup_(std::signal(SIGHUP, receiveSignal)), pipe_(std::signal(SIGPIPE, receiveSignal)) {}
            /** @brief 恢复旧处理器，无参数无前提无返回。 */
            ~TerminalSignalGuard() {
                std::signal(SIGINT, interrupt_);
                std::signal(SIGTERM, termination_);
                std::signal(SIGHUP, hangup_);
                std::signal(SIGPIPE, pipe_);
            }
            TerminalSignalGuard(const TerminalSignalGuard&) = delete; ///< 不复制全局资源。
            TerminalSignalGuard& operator=(const TerminalSignalGuard&) = delete; ///< 不赋值。
        private:
            using Handler = void (*)(int); ///< 信号处理器类型。
            Handler interrupt_, termination_, hangup_, pipe_; ///< 先前处理器。
        };
        /** @brief 移除控制字节并按UTF8边界限长。@param source 不可信显示文本。
         * @param limit 最大字节。@pre 无。@return 清理后的有界文本。
         * @code auto s=displayText(path,256); @endcode */
        std::string displayText(std::string_view source, std::size_t limit = 512) {
            std::string result;
            for (std::size_t i = 0; i < source.size() && result.size() < limit;) {
                const auto c = static_cast<unsigned char>(source[i]);
                if (c < 32 || c == 127) {
                    result += ' ';
                    ++i;
                    continue;
                }
                const std::size_t length = c < 128 ? 1
                    : c >= 194 && c <= 223         ? 2
                    : c >= 224 && c <= 239         ? 3
                    : c >= 240 && c <= 244         ? 4
                                                   : 0;
                bool valid = length && i + length <= source.size();
                for (std::size_t j = 1; valid && j < length; ++j) {
                    valid = (static_cast<unsigned char>(source[i + j]) & 192) == 128;
                }
                if (valid && length == 3) {
                    valid = !(c == 224 && static_cast<unsigned char>(source[i + 1]) < 160) &&
                        !(c == 237 && static_cast<unsigned char>(source[i + 1]) >= 160);
                }
                if (valid && length == 4) {
                    valid = !(c == 240 && static_cast<unsigned char>(source[i + 1]) < 144) &&
                        !(c == 244 && static_cast<unsigned char>(source[i + 1]) >= 144);
                }
                if (!valid) {
                    result += '?';
                    ++i;
                    continue;
                }
                if (result.size() + length > limit) {
                    break;
                }
                result.append(source.substr(i, length));
                i += length;
            }
            return result;
        }
        /** @brief 当前CUDA环境，查询失败字段留空不影响求解。@code EnvironmentInfo e; @endcode */
        struct EnvironmentInfo {
            std::string gpu_, runtime_, driverApi_, memory_; ///< GPU及版本、设备整体显存使用。
            int device_{}; ///< 计算线程必须选择的设备。
        };
        /** @brief 查询环境，无参数。@pre 当前CUDA设备有效。@return 展示信息，设备不可用抛异常。
         * @code auto environment=queryEnvironment(); @endcode */
        EnvironmentInfo queryEnvironment() {
            EnvironmentInfo info;
            if (cudaGetDevice(&info.device_) != cudaSuccess) {
                throw std::runtime_error("无法选择CUDA设备");
            }
            cudaDeviceProp prop{};
            if (cudaGetDeviceProperties(&prop, info.device_) == cudaSuccess) {
                info.gpu_ = displayText(prop.name) + " / sm_" + std::to_string(prop.major) + std::to_string(prop.minor);
            }
            const auto version = [](int n) { return std::to_string(n / 1000) + "." + std::to_string(n % 1000 / 10); };
            int value = 0;
            if (cudaRuntimeGetVersion(&value) == cudaSuccess) {
                info.runtime_ = version(value);
            }
            if (cudaDriverGetVersion(&value) == cudaSuccess) {
                info.driverApi_ = version(value);
            }
            std::size_t free = 0, total = 0;
            if (cudaMemGetInfo(&free, &total) == cudaSuccess) {
                std::ostringstream stream;
                stream << std::fixed << std::setprecision(2) << double(total - free) / (1ull << 30) << " / "
                       << double(total) / (1ull << 30) << " GiB";
                info.memory_ = stream.str();
            }
            return info;
        }
        /** @brief 只保存当前标量及最后8条消息，不保存逐迭代队列。@code TuiState state; @endcode */
        struct TuiState {
            problem::RunStage stage_{problem::RunStage::Loading}; ///< 当前阶段。
            std::string name_, mesh_, analysis_, solver_; ///< 拥有的算例摘要。
            std::size_t nodes_{}, cells_{}, dofs_{}, nonZeros_{}, step_{}, committed_{}, total_{}; ///< 规模及实际步数。
            std::optional<double> time_, dt_, acceptedResidual_; ///< 当前时间与步长。
            double relativeTolerance_{}, absoluteTolerance_{}, elapsedS_{}; ///< 配置及最终耗时。
            ConvergenceProgress convergence_; ///< 当前apply的收敛展示。
            std::optional<problem::ProgressInfo> progress_; ///< 当前数值组装段。
            std::deque<std::string> messages_; ///< 最多8条，每条512字节。
            bool finished_{}; ///< 数值资源已经释放。
            int code_{1}; ///< 最终退出码。
        };
        /** @brief 回调只更新有界状态，主线程定时复制快照。
         * @code TuiPresenter presenter; @endcode */
        class TuiPresenter final : public problem::RunObserver {
        public:
            /** @brief 覆盖当前迭代快照。@param info 临时标量。@pre 无。
             * @note 无返回值。@code presenter.onIteration(info); @endcode */
            void onIteration(const IterationInfo<double>& info) noexcept override {
                try {
                    std::lock_guard lock(mutex_);
                    state_.convergence_.update(info);
                }
                catch (...) {
                }
            }
            /** @brief 覆盖当前批计数。@param info 临时计数。@pre 无。
             * @note 无返回值。@code presenter.onProgress(info); @endcode */
            void onProgress(const problem::ProgressInfo& info) noexcept override {
                try {
                    std::lock_guard lock(mutex_);
                    state_.progress_ = info;
                }
                catch (...) {
                }
            }
            /** @brief 立即复制运行事件。@param event 临时视图。@pre 无。
             * @note 无返回值。@code presenter.onEvent(event); @endcode */
            void onEvent(const problem::RunEvent& event) noexcept override {
                try {
                    std::lock_guard lock(mutex_);
                    using problem::RunEventKind;
                    if (plain_) {
                        console_.onEvent(event);
                    }
                    switch (event.kind_) {
                    case RunEventKind::Stage:
                        state_.stage_ = event.stage_;
                        state_.progress_.reset();
                        break;
                    case RunEventKind::Metadata:
                        if (event.summary_) {
                            const auto& s = *event.summary_;
                            state_.name_ = displayText(s.name_);
                            state_.mesh_ = displayText(s.mesh_);
                            state_.analysis_ = displayText(s.analysis_);
                            state_.solver_ = displayText(s.solver_);
                            state_.nodes_ = s.nodes_;
                            state_.cells_ = s.cells_;
                            state_.dofs_ = s.dofs_;
                            state_.nonZeros_ = s.nonZeros_;
                            state_.relativeTolerance_ = s.relativeTolerance_;
                            state_.absoluteTolerance_ = s.absoluteTolerance_;
                        }
                        state_.total_ = event.totalSteps_;
                        break;
                    case RunEventKind::StepStarted:
                        state_.step_ = event.step_;
                        state_.time_ = event.time_;
                        state_.dt_ = event.dt_;
                        state_.convergence_.reset();
                        state_.acceptedResidual_.reset();
                        state_.progress_.reset();
                        break;
                    case RunEventKind::SolveAccepted:
                        state_.convergence_.accept();
                        state_.acceptedResidual_ = event.acceptedResidual_;
                        break;
                    case RunEventKind::StepCommitted:
                        state_.committed_ = event.step_;
                        break;
                    case RunEventKind::Message:
                        state_.messages_.push_back(displayText(event.message_));
                        if (state_.messages_.size() > 8) {
                            state_.messages_.pop_front();
                        }
                        break;
                    case RunEventKind::Finished:
                        state_.stage_ = event.stage_;
                        state_.elapsedS_ = event.elapsedS_;
                        state_.code_ = event.exitCode_;
                        state_.finished_ = true;
                        break;
                    }
                }
                catch (...) { /* 展示资源不足不影响求解。 */
                }
            }
            /** @brief 获取拥有型快照，无参数。@pre 无。@return 一致状态副本。
             * @code auto s=presenter.snapshot(); @endcode */
            TuiState snapshot() const {
                std::lock_guard lock(mutex_);
                return state_;
            }
            /** @brief 终端恢复后切普通输出，无参数。@pre 已退出全屏。@note 无返回值。
             * @code presenter.usePlain(); @endcode */
            void usePlain() {
                std::lock_guard lock(mutex_);
                plain_ = true;
                for (const auto& text : state_.messages_) {
                    std::cout << text << '\n';
                }
            }

        private:
            mutable std::mutex mutex_; ///< 短期保护标量及文本，不在锁内执行CUDA。
            TuiState state_; ///< 有界共享快照。
            bool plain_{}; ///< 恢复终端后允许逐行输出。
            ConsolePresenter console_; ///< 降级文本呈现。
        };
        /** @brief 计算线程始终在呈现资源销毁前结束。@code WorkerGuard guard(worker,control); @endcode */
        struct WorkerGuard {
            std::thread& worker_; ///< 借用线程。
            problem::RunControl& control_; ///< 借用取消器。
            ~WorkerGuard() {
                if (worker_.joinable()) {
                    control_.requestCancel();
                    worker_.join();
                }
            } ///< 异常也先请求再等待。
        };
        /** @brief 阶段英文标签。@param stage 阶段。@pre 有效枚举。@return 静态文本。
         * @code auto label=stageName(stage); @endcode */
        std::string stageName(problem::RunStage stage) {
            static const char* names[] = {"Loading case & mesh",
                                          "Preparing temperature space",
                                          "Assembling thermal model",
                                          "Building linear system",
                                          "Setting up solver",
                                          "Solving",
                                          "Checking true residual",
                                          "Postprocessing",
                                          "Saving results",
                                          "Finalizing archive",
                                          "Completed",
                                          "Failed",
                                          "Cancelled"};
            return names[static_cast<int>(stage)];
        }
        /** @brief 格式化标量。@param value 标量。@pre 无。@return 科学计数文本。
         * @code auto s=number(residual); @endcode */
        std::string number(double value) {
            std::ostringstream s;
            s << std::scientific << std::setprecision(3) << value;
            return s.str();
        }
        /** @brief 将快照构造成自适应布局，不读取CUDA或共享状态。
         * @param s 快照。@param e 环境。@param input 算例路径。@param output 输出路径。
         * @param caps 当前尺寸。@param seconds 已用秒。@param cancelling 取消已请求。
         * @pre 主线程调用。@return FTXUI文档。@code auto document=renderTui(s,e,input,output,caps,1,false); @endcode */
        ftxui::Element renderTui(const TuiState& s, const EnvironmentInfo& e, const std::string& input,
                                 const std::string& output, TerminalCapabilities caps, double seconds,
                                 bool cancelling) {
            using namespace ftxui;
            const auto orange = Color::RGB(255, 165, 76), sand = Color::RGB(214, 181, 145),
                       muted = Color::RGB(164, 158, 151);
            const auto row = [&](const std::string& label, const std::string& value) {
                return hbox(
                    {text(label) | color(muted) | size(WIDTH, EQUAL, 13), text(value.empty() ? "-" : value) | flex});
            };
            Elements title;
            if (caps.rows_ >= 34 && caps.columns_ >= 85) {
                title = {text("  ███████╗███╗   ███╗██████╗ ███████╗██████╗ ███████╗██╗███████╗██╗     ██████╗ "),
                         text("  ██╔════╝████╗ ████║██╔══██╗██╔════╝██╔══██╗██╔════╝██║██╔════╝██║     ██╔══██╗"),
                         text("  █████╗  ██╔████╔██║██████╔╝█████╗  ██████╔╝█████╗  ██║█████╗  ██║     ██║  ██║"),
                         text("  ██╔══╝  ██║╚██╔╝██║██╔══██╗██╔══╝  ██╔══██╗██╔══╝  ██║██╔══╝  ██║     ██║  ██║"),
                         text("  ███████╗██║ ╚═╝ ██║██████╔╝███████╗██║  ██║██║     ██║███████╗███████╗██████╔╝")};
            }
            else {
                title = {text(caps.columns_ < 60 ? "EMBERFIELD / Heat solver"
                                                 : "EMBERFIELD / CUDA finite-element heat solver") |
                         bold};
            }
            auto environment = window(text(" ENVIRONMENT ") | color(orange),
                                      vbox({row("GPU", e.gpu_), row("CUDA runtime", e.runtime_),
                                            row("Driver API", e.driverApi_), row("Memory*", e.memory_)}));
            auto details = window(
                text(" CASE ") | color(orange),
                vbox(
                    {row("Name", s.name_.empty() ? input : s.name_), row("Analysis", s.analysis_), row("Mesh", s.mesh_),
                     row("Size",
                         s.total_ ? std::to_string(s.nodes_) + " nodes / " + std::to_string(s.cells_) + " cells" : "-"),
                     row("System",
                         s.total_ ? std::to_string(s.dofs_) + " DOF / " + std::to_string(s.nonZeros_) + " nnz" : "-"),
                     row("Solver", s.solver_)}));
            auto panels = caps.columns_ >= 85
                ? hbox({environment | size(WIDTH, EQUAL, std::min(caps.columns_, 112) / 2 - 1), details | flex})
                : vbox({details});
            if (caps.rows_ < 28) {
                panels = row("Case", s.name_.empty() ? input : s.name_);
            }
            const char* spinner[] = {"◐", "◓", "◑", "◒"};
            Elements progress = {
                hbox({text(s.finished_ ? "● " : spinner[static_cast<int>(seconds * 10) % 4]) | color(orange),
                      text(" " + stageName(s.stage_)) | bold, filler(),
                      text(number(s.finished_ ? s.elapsedS_ : seconds) + " s")})};
            if (cancelling && !s.finished_) {
                progress.push_back(text("Cancellation requested; waiting for a safe boundary") | color(orange));
            }
            if (s.total_) {
                progress.push_back(
                    row("Run", std::to_string(s.committed_) + " / " + std::to_string(s.total_) + " committed steps"));
                progress.push_back(gauge(float(double(s.committed_) / s.total_)) | color(sand));
            }
            if (s.time_) {
                progress.push_back(row("Time", number(*s.time_) + " s / dt=" + (s.dt_ ? number(*s.dt_) : "-") + " s"));
            }
            if (s.stage_ == problem::RunStage::Assembling && s.progress_) {
                const auto& p = *s.progress_;
                static const char* works[] = {"Volume K/M", "Boundary terms", "Volume source", "Flux terms",
                                              "Convection"};
                progress.push_back(row(works[static_cast<int>(p.work_)],
                                       p.determinate_ ? std::to_string(p.completed_) + " / " + std::to_string(p.total_)
                                                      : "Finalizing assembly"));
                if (p.determinate_ && p.total_) {
                    progress.push_back(gauge(float(double(p.completed_) / p.total_)) | color(orange));
                }
            }
            const auto& c = s.convergence_;
            progress.push_back(row("Iteration", std::to_string(c.iteration()) + " / residual-based progress"));
            progress.push_back(row("Residual",
                                   c.residual() ? number(*c.residual()) +
                                           (c.residualKind() == ResidualKind::True ? " (true)" : " (recursive)")
                                                : "waiting for first iteration"));
            progress.push_back(row("Tolerance", c.tolerance() ? number(*c.tolerance()) : "-"));
            if (c.fraction()) {
                progress.push_back(
                    hbox({gauge(float(*c.fraction())) | flex | color(c.accepted() ? Color::Green : orange),
                          text(" " + std::to_string(static_cast<int>(*c.fraction() * 100)) + "%" +
                               (c.accepted() ? " accepted" : ""))}));
            }
            if (s.acceptedResidual_ && caps.rows_ >= 28) {
                progress.push_back(row("Accepted |r|", number(*s.acceptedResidual_)));
            }
            Elements logs;
            const auto lines = caps.rows_ >= 34 ? 4u : 1u;
            for (auto it = s.messages_.end() - std::min<std::size_t>(lines, s.messages_.size());
                 it != s.messages_.end(); ++it) {
                logs.push_back(text(*it) | color(muted));
            }
            Elements body = {vbox(title) | color(orange),
                             panels,
                             window(text(" PROGRESS ") | color(orange), vbox(progress)),
                             vbox(logs) | flex,
                             row("Results", output),
                             text(s.finished_ ? "Enter / Q / Esc  close" : "Q / Esc / Ctrl-C  cancel safely") |
                                 color(sand)};
            if (caps.rows_ >= 34) {
                body.push_back(text("* Device-wide memory at startup; includes other processes") | color(muted));
            }
            return vbox(body) | size(WIDTH, LESS_THAN, 112) | center | color(sand);
        }
    } // namespace
#endif
    int runTui(const std::filesystem::path& input, const std::filesystem::path& output,
               const problem::ResourceLimits& limits, bool noWait) {
#ifndef EMBERFIELD_BUILD_TUI
        (void)input;
        (void)output;
        (void)limits;
        (void)noWait;
        throw std::runtime_error("此构建未启用TUI");
#else
        using namespace ftxui;
        terminalSignal = 0;
        auto environment = queryEnvironment();
        const auto inputText = displayText(input.string()), outputText = displayText(output.string());
        TuiPresenter presenter;
        problem::RunControl control;
        std::exception_ptr failure;
        std::thread worker;
        WorkerGuard guard{worker, control};
        bool fallback = false;
        const auto started = Clock::now();
        {
            auto screen = ScreenInteractive::Fullscreen();
            screen.TrackMouse(false);
            screen.ForceHandleCtrlC(false);
            const auto renderer = Renderer([&] {
                return renderTui(presenter.snapshot(), environment, inputText, outputText, terminalCapabilities(),
                                 std::chrono::duration<double>(Clock::now() - started).count(), control.cancelled());
            });
            auto component = CatchEvent(renderer, [&](Event event) {
                if (event == Event::Character('q') || event == Event::Character('Q') || event == Event::Escape ||
                    event == Event::CtrlC || event == Event::Return) {
                    if (presenter.snapshot().finished_) {
                        screen.Exit();
                    }
                    else if (event != Event::Return) {
                        control.requestCancel();
                    }
                    return true;
                }
                return false;
            });
            Loop loop(&screen, component);
            TerminalSignalGuard signals;
            worker = std::thread([&] {
                try {
                    const auto error = cudaSetDevice(environment.device_);
                    if (error != cudaSuccess) {
                        throw std::runtime_error("计算线程选择CUDA设备失败");
                    }
                    problem::runCase(input, output, nullptr, limits, &presenter, &control);
                }
                catch (...) {
                    failure = std::current_exception();
                    presenter.onEvent({.kind_ = problem::RunEventKind::Finished,
                                       .stage_ = problem::RunStage::Failed,
                                       .exitCode_ = 1});
                }
            });
            while (!loop.HasQuitted()) {
                const auto state = presenter.snapshot();
                if (terminalSignal) {
                    control.requestCancel();
                    if (state.finished_) {
                        break;
                    }
                }
                pollfd descriptor{STDIN_FILENO, POLLHUP, 0};
                if (poll(&descriptor, 1, 0) > 0 && (descriptor.revents & (POLLHUP | POLLERR | POLLNVAL))) {
                    control.requestCancel();
                    if (state.finished_) {
                        break;
                    }
                }
                const auto caps = terminalCapabilities();
                if (caps.columns_ < 40 || caps.rows_ < 18) {
                    fallback = true;
                    break;
                }
                if (state.finished_ && (noWait || control.cancelled())) {
                    break;
                }
                screen.PostEvent(Event::Custom);
                loop.RunOnce();
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            if (loop.HasQuitted() && !presenter.snapshot().finished_) {
                control.requestCancel();
            }
        }
        TerminalSignalGuard waitingSignals;
        if (fallback) {
            std::cout << "[提示] 终端过小，继续使用普通输出\n";
            presenter.usePlain();
            while (!presenter.snapshot().finished_) {
                if (terminalSignal) {
                    control.requestCancel();
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
        if (worker.joinable()) {
            worker.join();
        }
        if (failure) {
            std::rethrow_exception(failure);
        }
        const auto final = presenter.snapshot();
        std::cout << "["
                  << (final.code_ == 0         ? "完成"
                          : final.code_ == 130 ? "已取消"
                                               : "失败")
                  << "] " << output.string() << '\n';
        if (final.code_ != 0 && !fallback) {
            for (const auto& message : final.messages_) {
                std::cout << message << '\n';
            }
        }
        return final.code_;
#endif
    }
} // namespace emberfield::cli
