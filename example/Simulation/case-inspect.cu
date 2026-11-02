/** @file case-inspect.cu
 * @brief 展示独立配置加载、MSH读取与算例绑定，输出阶段诊断而不伪装执行求解。
 */
#include <IO/Case/Case.cuh>
#include <IO/Mesh/Msh.cuh>
#include <Simulation/Case/Binding.cuh>
#include <exception>
#include <iostream>
#include <memory>
namespace {
    /** @brief 输出带字段位置的诊断。@param report 诊断集合。@pre 无。
     * @note 无返回值。@code printDiagnostics(result.diagnostics_); @endcode */
    void printDiagnostics(const emberfield::problem::Diagnostics& report) {
        for (const auto& d : report.entries()) {
            std::cerr << (d.severity_ == emberfield::problem::Severity::Error ? "错误" : "警告") << " [" << d.code_
                      << "] " << d.path_ << ": " << d.message_;
            if (!d.source_.empty()) {
                std::cerr << " (" << d.source_ << ":" << d.line_.value_or(1) << ":" << d.column_.value_or(1) << ")";
            }
            if (d.meshTag_) {
                std::cerr << " meshTag=" << *d.meshTag_;
            }
            std::cerr << '\n';
        }
        if (report.omitted()) {
            std::cerr << "省略诊断: " << report.omitted() << '\n';
        }
    }
} // namespace
/** @brief 检查一个完整配置与网格的绑定。
 * @param argc 参数数量。@param argv 配置文件路径。@pre 无。
 * @return 0表示配置和身份绑定通过，2表示输入/绑定失败，1表示系统异常；0不表示可求解。
 * @code ./case-inspect example/cases/steady.json @endcode */
int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "用法: case-inspect <case.json>\n";
        return 2;
    }
    try {
        auto input = emberfield::io::loadCase(argv[1]);
        printDiagnostics(input.diagnostics_);
        if (!input.definition_) {
            return 2;
        }
        auto document = emberfield::io::loadMsh(input.definition_->mesh_.path_);
        auto mesh = std::make_shared<const emberfield::mesh::Mesh>(std::move(document.mesh_));
        auto result = emberfield::problem::bindCase(*input.definition_, std::move(mesh));
        printDiagnostics(result.diagnostics_);
        if (!result.bound_) {
            return 2;
        }
        const auto& d = result.bound_->data();
        std::cout << "配置与网格身份绑定通过\n活动单元: " << d.active_.size()
                  << "\n材料: " << d.definition_.materials_.size() << "\n外侧默认绝热: " << d.insulatedFacets_.size()
                  << '\n';
        printDiagnostics(d.capability_.diagnostics_);
        for (const auto& pending : d.capability_.pending_) {
            std::cout << "待完成: " << pending.message_ << '\n';
        }
        std::cout << "可执行求解: " << (d.capability_.executable() ? "是" : "否") << '\n';
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
