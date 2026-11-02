/** @file System.cu
 * @brief 按编号分量诊断常数零空间，并调度独立GPU系统操作。
 */
#include <Simulation/Case/System.cuh>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
namespace thermofem::problem {
    bool SolvabilityReport::canBuildSystem() const noexcept {
        return !components_.empty() &&
            std::all_of(components_.begin(), components_.end(), [](const ComponentCheck& c) { return c.anchored(); });
    }
    SolvabilityReport checkThermalSolvability(const AssembledThermalModel& m, double relative, double absolute) {
        if (!std::isfinite(relative) || !std::isfinite(absolute) || relative < 0 || absolute < 0) {
            throw std::invalid_argument("载荷平衡容差必须非负有限");
        }
        const auto& plan = *m.plan()->plan();
        const auto& p = *m.plan()->prepared();
        SolvabilityReport r;
        r.components_.resize(plan.componentCount());
        const auto& labels = plan.components();
        for (std::size_t i = 0; i < r.components_.size(); ++i) {
            r.components_[i].component_ = i;
        }
        for (auto c : labels) {
            ++r.components_[c].dofCount_;
        }
        for (auto d : p.conditions()->constraints().dofs()) {
            r.components_[labels[d]].hasDirichlet_ = true;
        }
        for (const auto& t : p.conditions()->naturalTerms()) {
            const auto& v = p.bound()->data().definition_.boundaries_[t.definitionIndex()].value_;
            if (const auto* c = std::get_if<ConvectionBoundary>(&v); c && evaluate(c->coefficient_) > 0) {
                for (auto d : t.globalDofs()) {
                    r.components_[labels[d]].hasPositiveConvection_ = true;
                }
            }
        }
        std::vector<double> load(labels.size());
        m.load().copyToHost(load);
        for (std::size_t i = 0; i < labels.size(); ++i) {
            auto& c = r.components_[labels[i]];
            c.netLoad_ += static_cast<long double>(load[i]);
            c.absoluteLoad_ += std::abs(static_cast<long double>(load[i]));
        }
        for (auto& c : r.components_) {
            c.hasCapacity_ = m.capacity().has_value();
            if (!std::isfinite(c.netLoad_) || !std::isfinite(c.absoluteLoad_)) {
                throw std::overflow_error("分量功率求和溢出");
            }
            c.loadCompatible_ = std::abs(c.netLoad_) <= static_cast<long double>(absolute) + relative * c.absoluteLoad_;
        }
        return r;
    }
    namespace {
        /** @brief 拒绝未锚定系统并区分不唯一和不相容。@param m 模型。
         * @pre 无。@note 无返回值，未锚定domain_error。@code requireAnchors(m); @endcode */
        void requireAnchors(const AssembledThermalModel& m) {
            const auto r = checkThermalSolvability(m);
            for (const auto& c : r.components_) {
                if (!c.anchored()) {
                    throw std::domain_error(
                        "热分量" + std::to_string(c.component_) +
                        (c.loadCompatible_ ? "为平衡纯Neumann，温度不唯一" : "为不平衡纯Neumann，方程不相容"));
                }
            }
        }
    } // namespace
    systems::LinearSystem buildThermalSteadySystem(const AssembledThermalModel& m) {
        if (m.capacity()) {
            throw std::invalid_argument("稳态工厂需要稳态模型");
        }
        requireAnchors(m);
        const auto& c = m.plan()->prepared()->conditions()->constraints();
        return systems::applyDirichlet(systems::buildSteadySystem(m.spatial(), m.load()), c.dofs(), c.temperatures());
    }
    systems::LinearSystem buildThermalStepSystem(const AssembledThermalModel& m, const Vector<double>& previous,
                                                 double dt) {
        if (!m.capacity()) {
            throw std::invalid_argument("后向Euler工厂需要瞬态模型");
        }
        requireAnchors(m);
        const auto& c = m.plan()->prepared()->conditions()->constraints();
        if (previous.size() != m.spatial().rowCount()) {
            throw std::invalid_argument("上一时刻温度维度错误");
        }
        systems::validateTemperatureState(previous, c.dofs(), c.temperatures());
        return systems::applyDirichlet(
            systems::buildBackwardEulerSystem(m.spatial(), *m.capacity(), m.load(), previous, dt), c.dofs(),
            c.temperatures());
    }
    std::optional<TimeStep> nextTimeStep(const TimeOptions& t, double current, std::size_t completed) {
        if (!std::isfinite(t.start_) || !std::isfinite(t.end_) || !std::isfinite(t.step_) || t.end_ <= t.start_ ||
            t.step_ <= 0 || !t.maxSteps_ || !std::isfinite(current) || current < t.start_ || current > t.end_ ||
            completed > t.maxSteps_) {
            throw std::invalid_argument("时间推进参数错误");
        }
        if (current == t.end_) {
            return std::nullopt;
        }
        if (completed == t.maxSteps_) {
            throw std::length_error("时间推进超过maxSteps");
        }
        // 从起点与整数步号计算名义终点，避免反复加step积累时间漂移。
        const long double nominal = static_cast<long double>(t.start_) +
            static_cast<long double>(completed + 1) * static_cast<long double>(t.step_);
        double end = static_cast<double>(std::min(nominal, static_cast<long double>(t.end_)));
        // 合并名义终点舍入造成的微小余步，无法前进仍然报错。
        const double snap = 8 * std::numeric_limits<double>::epsilon() * std::max(std::abs(t.end_), std::abs(current));
        if (end > current && end < t.end_ && t.end_ - end <= snap) {
            end = t.end_;
        }
        const double duration = end - current;
        if (!std::isfinite(end) || !std::isfinite(duration) || end <= current || end > t.end_) {
            throw std::overflow_error("时间步无法有限且严格地推进");
        }
        return TimeStep{completed + 1, current, end, duration};
    }
} // namespace thermofem::problem
