/** @file Run.cuh @brief 从完整算例到结果目录的正式运行入口。 */
#pragma once
#include <Simulation/Case/ResourceLimits.cuh>
#include <Simulation/Case/RunObserver.cuh>
#include <csignal>
#include <filesystem>
namespace emberfield::problem {
    /** @brief 加载、准备、GPU组装并求解，按配置发布结果。
     * @param input 配置路径。@param output 最终输出目录，覆盖output.directory。
     * @param interrupted 可选信号标志，只读借用，非零时在安全步边界中断。
     * @param limits 组合的读取、空间、准备、组装和输出预算；默认保留模块原生值。
     * @pre 当前CUDA设备可用，来源文件可读。@return 0成功，2输入失败，3数值失败，130中断，1资源或输出错误。
     * @note 会话建立前错误写stderr，其后失败保留已提交结果。
     * @code auto code=runCase("case.json","results/case"); @endcode */
    int runCase(const std::filesystem::path& input, const std::filesystem::path& output,
                const volatile std::sig_atomic_t* interrupted = nullptr, const ResourceLimits& limits = {});
    /** @brief 使用指定呈现端运行，保留原重载的退出及归档语义。
     * @param input 输入路径。@param output 输出目录。@param interrupted 可空信号标志。
     * @param limits 资源预算。@param observer 可空借用观察者，空时使用普通文本。
     * @param control 可空原子取消标志。@pre 同原入口，观察者和控制器覆盖本调用。
     * @return 与原入口相同，Finished在数值资源释放后通知。
     * @code auto code=runCase(input,output,nullptr,limits,&observer,&control); @endcode */
    int runCase(const std::filesystem::path& input, const std::filesystem::path& output,
                const volatile std::sig_atomic_t* interrupted, const ResourceLimits& limits, RunObserver* observer,
                RunControl* control);
} // namespace emberfield::problem
