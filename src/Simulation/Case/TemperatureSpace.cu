/** @file TemperatureSpace.cu
 * @brief 实现算例到通用空间的单向适配，区分编号成功与物理检查完成。
 */
#include <Simulation/Case/TemperatureSpace.cuh>
#include <stdexcept>

namespace thermofem::problem {
    TemperatureSpaceResult buildTemperatureSpace(const BoundCase& bound, fem::SpaceLimits limits) {
        if (!limits.maxCells_ || !limits.maxLocalDofs_ || !limits.maxGlobalDofs_ || !limits.maxDiagnostics_) {
            throw std::invalid_argument("温度空间限额必须为正");
        }
        TemperatureSpaceResult result{{}, Diagnostics(limits.maxDiagnostics_)};
        const auto& data = bound.data();
        // 未定义周期温度策略前保守拒绝整个含周期关系的网格，不猜测哪些关系物理有效。
        if (!data.mesh_->data().periodic_.empty()) {
            result.diagnostics_.add({Severity::Error,
                                     "SPACE_PERIODIC_POLICY_REQUIRED",
                                     "/mesh",
                                     {},
                                     {},
                                     "网格含周期关系；首期温度空间适配未定义周期物理策略",
                                     "显式实现周期条件策略后再准备计算"});
            return result;
        }
        if (data.active_.size() > limits.maxCells_) {
            result.diagnostics_.add(
                {Severity::Error, "SPACE_RESOURCE_LIMIT", "/discretization", {}, {}, "活动单元数超过空间限额"});
            return result;
        }
        const int dimension = data.definition_.analysis_.space_ == SpaceModel::Planar2d ? 2 : 3;
        std::vector<fem::ElementKey> keys;
        keys.reserve(data.active_.size());
        for (auto cell : data.active_) {
            const auto& block = data.mesh_->data().elements_.at(cell.block_);
            if (block.type_.dimension_ != dimension) {
                result.diagnostics_.add({Severity::Error,
                                         "SPACE_DIMENSION_MISMATCH",
                                         "/analysis/dimension",
                                         {},
                                         block.tags_.at(cell.element_),
                                         "分析维数与活动单元维数不一致"});
            }
            keys.push_back(
                {block.type_.family_, fem::BasisFamily::ContinuousLagrange, data.definition_.discretization_.order_});
        }
        if (result.diagnostics_.hasErrors()) {
            return result;
        }
        auto built = fem::buildFiniteElementSpace(data.mesh_, data.active_, keys, limits);
        for (const auto& issue : built.issues_) {
            std::optional<std::uint64_t> tag;
            if (issue.cell_ && issue.cell_->block_ < data.mesh_->data().elements_.size()) {
                const auto& tags = data.mesh_->data().elements_[issue.cell_->block_].tags_;
                if (issue.cell_->element_ < tags.size()) {
                    tag = tags[issue.cell_->element_];
                }
            }
            result.diagnostics_.add({issue.error_ ? Severity::Error : Severity::Warning,
                                     fem::spaceIssueName(issue.code_),
                                     "/discretization",
                                     {},
                                     tag,
                                     issue.message_});
        }
        if (built.truncated_) {
            result.diagnostics_.add({built.hasErrors_ ? Severity::Error : Severity::Warning,
                                     "SPACE_DIAGNOSTICS_TRUNCATED",
                                     "/discretization",
                                     {},
                                     {},
                                     "底层空间诊断已截断，保留完整失败状态"});
        }
        if (built.hasErrors_ && !result.diagnostics_.hasErrors()) {
            result.diagnostics_.add({Severity::Error, "SPACE_BUILD_FAILED", "/discretization", {}, {}, "空间构建失败"});
        }
        if (!result.diagnostics_.hasErrors()) {
            result.space_ = std::move(built.space_);
        }
        return result;
    }
} // namespace thermofem::problem
