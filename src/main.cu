/**
 * @file main.cu
 * @brief CLI11命令行入口，解析路径和资源策略并展示运行环境。
 */

#include <CLI11.hpp>
#include <IO/Case/CommandLine.cuh>
#include <IO/Case/ResourceLimits.cuh>
#include <NVTXProfiler.cuh>
#include <Simulation/Case/Run.cuh>
#include <version.cuh>

#include <csignal>
#include <cuda_runtime_api.h>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>

namespace {
    volatile std::sig_atomic_t interrupted = 0; ///< 仅信号处理器写，安全步边界读。
    /** @brief 将CUDA整数版本转换为主次版本文本。
     * @param value CUDA版本整数。@pre value非负。@return 主次版本字符串。
     * @code auto text=cudaVersion(12080); @endcode */
    std::string cudaVersion(int value) {
        return std::to_string(value / 1000) + "." + std::to_string((value % 1000) / 10);
    }
    /** @brief 输出当前CUDA设备和运行时环境，无参数。
     * @pre 只在普通运行参数解析后调用。@note 无返回值；设备查询失败仅影响展示。
     * @code printHardware(); @endcode */
    void printHardware() {
        EMBERFIELD_NVTX_SCOPE_COLOR("printHardware", emberfield::NvtxColor::Rose);
        int device = 0;
        cudaDeviceProp properties{};
        if (cudaGetDevice(&device) == cudaSuccess && cudaGetDeviceProperties(&properties, device) == cudaSuccess) {
            std::ostringstream description;
            description << properties.name << " (sm_" << properties.major << properties.minor << ", " << std::fixed
                        << std::setprecision(1)
                        << static_cast<double>(properties.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0) << " GiB)";
            std::cout << "  GPU          " << description.str() << '\n';
        }
        else {
            std::cout << "  GPU          未能查询当前设备\n";
        }
        int runtime = 0, driver = 0;
        if (cudaRuntimeGetVersion(&runtime) == cudaSuccess && cudaDriverGetVersion(&driver) == cudaSuccess) {
            std::cout << "  CUDA         runtime " << cudaVersion(runtime) << " / driver " << cudaVersion(driver)
                      << '\n';
        }
    }
    /** @brief 输出一次性程序标识、设备和路径摘要。
     * @param version 程序版本。@param input 算例路径。@param output 输出目录。
     * @param limits 可选资源文件路径。@pre CLI参数已通过解析与路径校验。
     * @note 无返回值，后续详细算例信息由运行入口展示。
     * @code printBanner(version,"case.json","results/case",""); @endcode */
    void printBanner(const emberfield::Version& version, const std::string& input, const std::filesystem::path& output,
                     const std::string& limits) {
        EMBERFIELD_NVTX_SCOPE_COLOR("printBanner", emberfield::NvtxColor::Rose);
        std::cout << R"(
  +-----------------------------------------------------------+
  |  E M B E R F I E L D                                      |
  |  CUDA finite-element heat solver                          |
  +-----------------------------------------------------------+
)";
        std::cout << "  版本         " << version.major_ << '.' << version.minor_ << '.' << version.patch_ << '\n';
        printHardware();
        std::cout << "  算例文件     " << input << '\n'
                  << "  结果目录     " << output.string() << '\n'
                  << "  资源配置     " << (limits.empty() ? "模块默认值" : limits) << "\n\n"
                  << std::flush;
    }
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
    EMBERFIELD_NVTX_SCOPE_COLOR("main", emberfield::NvtxColor::Rose);
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
        {
            EMBERFIELD_NVTX_SCOPE("emberfield::cli/parse");
            argv = app.ensure_utf8(argv);
            app.parse(argc, argv);
        }
        const auto limits = resourceLimits.empty() ? emberfield::problem::ResourceLimits{}
                                                   : emberfield::io::loadResourceLimits(resourceLimits);
        const auto directory = emberfield::io::outputDirectory(
            input, root.empty() ? emberfield::io::executableDirectory() : std::filesystem::path(root),
            output->count() ? std::optional<std::string>(name) : std::nullopt);
        printBanner(version, input, directory, resourceLimits);
        SignalGuard guard;
        return emberfield::problem::runCase(input, directory, &interrupted, limits);
    }
    catch (const CLI::ParseError& e) {
        return app.exit(e);
    }
    catch (const std::exception& e) {
        std::cerr << "[错误] " << e.what() << '\n';
        return 1;
    }
}
