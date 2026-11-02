/** @file Run.cuh @brief 从完整算例到结果目录的正式运行入口。 */
#pragma once
#include <csignal>
#include <filesystem>
namespace emberfield::problem {
    /** @brief 加载、准备、GPU组装并求解，按配置发布结果。
     * @param input 配置路径。@param output 最终输出目录，覆盖output.directory。
     * @param interrupted 可选信号标志，只读借用，非零时在安全步边界中断。
     * @pre 当前CUDA设备可用，来源文件可读。@return 0成功，2输入失败，3数值失败，130中断，1资源或输出错误。
     * @note 会话建立前错误写stderr，其后失败保留已提交结果。
     * @code auto code=runCase("case.json","results/case"); @endcode */
    int runCase(const std::filesystem::path& input, const std::filesystem::path& output,
                const volatile std::sig_atomic_t* interrupted = nullptr);
} // namespace emberfield::problem
