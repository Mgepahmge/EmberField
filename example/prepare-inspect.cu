/** @file prepare-inspect.cu
 * @brief 从完整算例检查几何、物理条件和流式主机/GPU局部贡献，不生成全局矩阵。
 */
#include <Case/GpuPreparation.cuh>
#include <IO/Case.cuh>
#include <IO/Msh.cuh>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <string>

namespace {
    /** @brief 输出限长诊断。@param d 诊断。@pre 无。@note 无返回值。
     * @code printDiagnostics(r.diagnostics()); @endcode */
    void printDiagnostics(const thermofem::problem::Diagnostics& d) {
        for (const auto& x : d.entries()) {
            std::cerr << x.code_ << ": " << x.message_ << '\n';
        }
        if (d.omitted()) {
            std::cerr << "省略诊断: " << d.omitted() << '\n';
        }
    }
    /** @brief 校验GPU实体结果与主机解析路径的数值一致性。
     * @param batch 回读数组。@param index 实体行。@param host 主机条目。@pre 形状相容。
     * @note 无返回值，数值差异超过相对/绝对门槛抛runtime_error。
     * @code compare(deviceValues,e,host.entries()); @endcode */
    void compare(const std::vector<double>& batch, std::size_t index, const std::vector<double>& host) {
        for (std::size_t i = 0; i < host.size(); ++i) {
            const double a = batch.at(index * host.size() + i), b = host[i];
            if (!std::isfinite(a) || std::abs(a - b) > 1e-10 * std::max(1.0, std::abs(b))) {
                throw std::runtime_error("GPU与主机局部贡献不一致");
            }
        }
    }
    /** @brief 流式求值并核对所有体项，不持久缓存全域稠密矩阵。
     * @param p 联合准备。@param gpu 是否GPU对照。@pre 来源有效。
     * @return 合计体源功率W。@code auto f=cells(p,true); @endcode */
    double cells(const thermofem::problem::PreparedThermalCase& p, bool gpu) {
        double load = 0;
        for (std::size_t begin = 0; begin < p.space()->cellCount(); begin += 4096) {
            const auto count = std::min<std::size_t>(4096, p.space()->cellCount() - begin);
            std::vector<std::size_t> rows(count);
            std::iota(rows.begin(), rows.end(), begin);
            std::optional<thermofem::problem::GpuThermalCellTerms> batch;
            std::vector<double> k, m, f;
            if (gpu) {
                batch = thermofem::problem::evaluateCellBatch(p, rows);
                k = batch->conductivity().copyToHost();
                f = batch->load().copyToHost();
                if (batch->capacity()) {
                    m = batch->capacity()->copyToHost();
                }
            }
            for (std::size_t e = 0; e < count; ++e) {
                const auto host = thermofem::problem::evaluateCellTerms(p, rows[e]);
                if (gpu) {
                    compare(k, e, host.conductivity().entries());
                    compare(f, e, host.load().entries());
                    if (host.capacity()) {
                        compare(m, e, host.capacity()->entries());
                    }
                }
                load = std::accumulate(host.load().entries().begin(), host.load().entries().end(), load);
            }
        }
        return load;
    }
    /** @brief 按类型流式核对自然侧项。@param p 联合上下文。@param gpu 是否GPU对照。
     * @pre 来源不变。@note 无返回，错误抛异常。@code boundaries(p,true); @endcode */
    void boundaries(const thermofem::problem::PreparedThermalCase& p, bool gpu) {
        using namespace thermofem;
        const auto& terms = p.conditions()->naturalTerms();
        for (bool flux : {true, false}) {
            std::vector<std::size_t> rows;
            const auto run = [&] {
                if (rows.empty()) {
                    return;
                }
                if (gpu) {
                    auto batch = problem::evaluateBoundaryBatch(p, rows);
                    if (flux) {
                        const auto values = std::get<thermal::GpuLocalVectorBatch>(batch.contribution()).copyToHost();
                        for (std::size_t e = 0; e < rows.size(); ++e) {
                            compare(
                                values, e,
                                std::get<thermal::LocalVector>(problem::evaluateBoundaryTerm(p, rows[e])).entries());
                        }
                    }
                    else {
                        const auto& c = std::get<thermal::GpuConvectionBatch>(batch.contribution());
                        const auto matrix = c.matrix().copyToHost(), load = c.load().copyToHost();
                        for (std::size_t e = 0; e < rows.size(); ++e) {
                            const auto host =
                                std::get<thermal::ConvectionContribution>(problem::evaluateBoundaryTerm(p, rows[e]));
                            compare(matrix, e, host.matrix().entries());
                            compare(load, e, host.load().entries());
                        }
                    }
                }
                else {
                    for (auto row : rows) {
                        (void)problem::evaluateBoundaryTerm(p, row);
                    }
                }
                rows.clear();
            };
            for (std::size_t i = 0; i < terms.size(); ++i) {
                const auto& value = p.bound()->data().definition_.boundaries_[terms[i].definitionIndex()].value_;
                if (std::holds_alternative<problem::FluxBoundary>(value) == flux) {
                    rows.push_back(i);
                }
                if (rows.size() == 4096) {
                    run();
                }
            }
            run();
        }
    }
} // namespace
/** @brief 完整读取配置并检查第4—6步。@param argc 参数数。@param argv 配置和可选cpu/gpu。
 * @pre 无。@return 0通过、2输入/准备失败、1运行异常；通过不代表全局可解。
 * @code ./prepare-inspect example/cases/transient.json gpu @endcode */
int main(int argc, char** argv) {
    using namespace thermofem;
    if (argc < 2 || argc > 3 || (argc == 3 && std::string(argv[2]) != "cpu" && std::string(argv[2]) != "gpu")) {
        std::cerr << "用法: prepare-inspect <case.json> [cpu|gpu]（默认gpu）\n";
        return 2;
    }
    try {
        const bool gpu = argc != 3 || std::string(argv[2]) == "gpu";
        auto input = io::loadCase(argv[1]);
        printDiagnostics(input.diagnostics_);
        if (!input.definition_) {
            return 2;
        }
        auto document = io::loadMsh(input.definition_->mesh_.path_);
        auto bound =
            problem::bindCase(*input.definition_, std::make_shared<const mesh::Mesh>(std::move(document.mesh_)));
        printDiagnostics(bound.diagnostics_);
        if (!bound.bound_) {
            return 2;
        }
        auto space = problem::buildTemperatureSpace(*bound.bound_);
        printDiagnostics(space.diagnostics_);
        if (!space.space_) {
            return 2;
        }
        auto result = problem::prepareThermalCase(bound.bound_, space.space_);
        printDiagnostics(result.diagnostics());
        if (!result.ok()) {
            return 2;
        }
        const auto& p = *result.prepared();
        double measure = 0;
        for (const auto& c : p.geometry()->cells()) {
            measure += c.measure();
        }
        const auto source = cells(p, gpu);
        boundaries(p, gpu);
        std::cout << "单元: " << p.space()->cellCount() << "，DOF: " << p.space()->dofCount() << "，体测度: " << measure
                  << "，规定温度DOF: " << p.conditions()->constraints().dofs().size()
                  << "，自然侧项: " << p.conditions()->naturalTerms().size() << "，体源功率W: " << source
                  << "，局部贡献检查: " << (gpu ? "GPU与CPU一致" : "CPU通过") << "\n";
        std::cout << "未进行全局组装、可解性认证或求解。\n";
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
