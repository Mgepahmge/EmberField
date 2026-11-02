/** @file space-inspect.cu
 * @brief 从算例配置建立并检查有限元空间，可验证MSH保存重载后的持久身份映射。
 */
#include <IO/Case/Case.cuh>
#include <IO/Mesh/Msh.cuh>
#include <Simulation/Case/TemperatureSpace.cuh>
#include <algorithm>
#include <exception>
#include <iostream>
#include <set>
#include <stdexcept>

namespace {
    /** @brief 输出诊断摘要。@param report 算例诊断。@pre 无。
     * @note 无返回值。@code printDiagnostics(r.diagnostics_); @endcode */
    void printDiagnostics(const emberfield::problem::Diagnostics& report) {
        for (const auto& d : report.entries()) {
            std::cerr << d.code_ << ": " << d.message_ << '\n';
        }
        if (report.omitted()) {
            std::cerr << "省略诊断: " << report.omitted() << '\n';
        }
    }
    /** @brief 以持久单元tag和局部顺序记录DOF关联的节点tag。
     * @param space 有效P1空间。@pre 每个DOF关联节点。
     * @return 记录数组；用于跨网格存储重排对比。@code auto ids=identities(space); @endcode */
    std::vector<std::pair<emberfield::mesh::Tag, std::vector<emberfield::mesh::Tag>>>
    identities(const emberfield::fem::FiniteElementSpace& space) {
        std::vector<std::pair<emberfield::mesh::Tag, std::vector<emberfield::mesh::Tag>>> result;
        for (std::size_t e = 0; e < space.cellCount(); ++e) {
            const auto cell = space.cells()[e];
            std::vector<emberfield::mesh::Tag> nodes;
            for (auto dof : space.cellDofs(e)) {
                nodes.push_back(space.mesh().data().nodes_.at(*space.dofInfo(dof).node_).tag_);
            }
            result.emplace_back(space.mesh().data().elements_[cell.block_].tags_[cell.element_], std::move(nodes));
        }
        std::sort(result.begin(), result.end());
        return result;
    }
} // namespace
/** @brief 读取配置并检查空间，可选将网格保存至指定路径后重载核对身份。
 * @param argc 参数数量。@param argv 配置路径及可选往返输出路径。@pre 无。
 * @return 0为空间检查通过（不表示可求解），2为输入/能力错误，1为异常或映射不一致。
 * @code ./space-inspect example/cases/steady.json build/space-roundtrip.msh @endcode */
int main(int argc, char** argv) {
    using namespace emberfield;
    if (argc != 2 && argc != 3) {
        std::cerr << "用法: space-inspect <case.json> [roundtrip.msh]\n";
        return 2;
    }
    try {
        const auto input = io::loadCase(argv[1]);
        printDiagnostics(input.diagnostics_);
        if (!input.definition_) {
            return 2;
        }
        auto document = io::loadMsh(input.definition_->mesh_.path_);
        auto mesh = std::make_shared<const mesh::Mesh>(document.mesh_);
        const auto binding = problem::bindCase(*input.definition_, mesh);
        printDiagnostics(binding.diagnostics_);
        if (!binding.bound_) {
            return 2;
        }
        const auto result = problem::buildTemperatureSpace(*binding.bound_);
        printDiagnostics(result.diagnostics_);
        if (!result.space_) {
            return 2;
        }
        const auto& space = *result.space_;
        // 独立收集活动顶点，核对映射语义，而非只读取构建器的数组长度。
        std::set<mesh::Index> activeNodes;
        for (std::size_t e = 0; e < space.cellCount(); ++e) {
            const auto cell = space.cells()[e];
            const auto nodes = mesh->connectivity(cell.block_, cell.element_);
            const auto row = space.cellDofs(e);
            if (row.size() != nodes.size()) {
                throw std::runtime_error("P1局部自由度数与顶点数不一致");
            }
            for (std::size_t a = 0; a < row.size(); ++a) {
                activeNodes.insert(nodes[a]);
                if (space.dofInfo(row[a]).node_ != nodes[a] || space.nodeDof(nodes[a]) != row[a]) {
                    throw std::runtime_error("局部到全局映射未保留节点身份");
                }
            }
        }
        if (activeNodes.size() != space.dofCount()) {
            throw std::runtime_error("连续P1全局DOF数与活动顶点数不一致");
        }
        if (argc == 3) {
            io::saveMsh(argv[2], document, io::MshEncoding::Ascii,
                        document.sourceVersion_ == "2.2" ? io::MshVersion::V22 : io::MshVersion::V41);
            auto loaded = io::loadMsh(argv[2]);
            auto rebound =
                problem::bindCase(*input.definition_, std::make_shared<const mesh::Mesh>(std::move(loaded.mesh_)));
            if (!rebound.bound_) {
                throw std::runtime_error("MSH保存重载后身份绑定失败");
            }
            auto rebuilt = problem::buildTemperatureSpace(*rebound.bound_);
            if (!rebuilt.space_ || identities(space) != identities(*rebuilt.space_)) {
                throw std::runtime_error("MSH保存重载后的自由度持久身份关联不一致");
            }
            std::cout << "MSH保存重载后局部DOF身份一致\n";
        }
        std::cout << "有限元空间编号通过\n活动单元: " << space.cellCount() << "\n全局DOF: " << space.dofCount()
                  << "\n仅完成未约束空间，几何、物理条件、组装与输出仍待完成\n";
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
