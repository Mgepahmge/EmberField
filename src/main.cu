/**
 * @file main.cu
 * @brief CLI11命令行入口，接受算例、输出路径和统一资源策略文件。
 */

#include <CLI11.hpp>
#include <IO/Case/CommandLine.cuh>
#include <IO/Case/ResourceLimits.cuh>
#include <Simulation/Case/Run.cuh>
#include <version.cuh>

#include <csignal>
#include <iostream>

namespace {
    volatile std::sig_atomic_t interrupted = 0; ///< 仅信号处理器写，安全步边界读。
    /** @brief 信号处理器只设标志。@param signal 信号编号。@pre 无。
     * @note 无返回值，不访问文件或CUDA。@code handleInterrupt(SIGINT); @endcode */
    void handleInterrupt(int signal) { interrupted = signal; }
    /** @brief RAII恢复主程序借用的信号处理器。@code SignalGuard guard; @endcode */
    class SignalGuard {
    public:
        /** @brief 保存并注册中断处理器，无参数无前提。@note 无返回值。@code SignalGuard guard; @endcode */
        SignalGuard() :
            interrupt_(std::signal(SIGINT, handleInterrupt)), termination_(std::signal(SIGTERM, handleInterrupt)) {}
        ~SignalGuard() noexcept {
            if (interrupt_ != SIG_ERR) {
                std::signal(SIGINT, interrupt_);
            }
            if (termination_ != SIG_ERR) {
                std::signal(SIGTERM, termination_);
            }
        } ///< 恢复原处理器。
        SignalGuard(const SignalGuard&) = delete; ///< 不复制全局资源。
        SignalGuard& operator=(const SignalGuard&) = delete; ///< 不复制全局资源。
    private:
        using Handler = void (*)(int); ///< 信号回调类型。
        Handler interrupt_, termination_; ///< 原处理器。
    };
} // namespace

/**
 * @brief 解析参数并运行完整算例。
 * @param argc 参数数。@param argv UTF-8参数。
 * @pre 运行算例要求CUDA可用；--help/--version不初始化CUDA。
 * @return 0成功，解析或求解失败为非零。
 * @code{.sh}
 * ./build/emberfield case.json --output-root results --output plate
 * @endcode
 */
int main(int argc, char** argv) {
    CLI::App app{"EmberField 热传导有限元求解器"};
    std::string input, root, name, resourceLimits;
    app.add_option("input", input, "算例配置文件路径")->required()->check(CLI::ExistingFile);
    app.add_option("--output-root", root, "结果目录的保存位置，默认可执行文件目录");
    auto* output = app.add_option("--output", name, "结果目录名，默认算例配置文件名（去扩展名）");
    app.add_option("--resource-limits", resourceLimits, "分阶段资源限制JSON文件路径")->check(CLI::ExistingFile);
    const auto version = emberfield::version();
    app.set_version_flag("--version",
                         "EmberField " + std::to_string(version.major_) + "." + std::to_string(version.minor_) + "." +
                             std::to_string(version.patch_));
    try {
        argv = app.ensure_utf8(argv);
        app.parse(argc, argv);
        const auto limits = resourceLimits.empty() ? emberfield::problem::ResourceLimits{}
                                                   : emberfield::io::loadResourceLimits(resourceLimits);
        const auto directory = emberfield::io::outputDirectory(
            input, root.empty() ? emberfield::io::executableDirectory() : std::filesystem::path(root),
            output->count() ? std::optional<std::string>(name) : std::nullopt);
        SignalGuard guard;
        return emberfield::problem::runCase(input, directory, &interrupted, limits);
    }
    catch (const CLI::ParseError& e) {
        return app.exit(e);
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
