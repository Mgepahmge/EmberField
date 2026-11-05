/** @file Assembly.cu
 * @brief 按固定体/侧顺序调用独立GPU积分和通用散加接口。
 */
#include <NVTXProfiler.cuh>
#include <Simulation/Case/Assembly.cuh>
#include <algorithm>
#include <numeric>
#include <stdexcept>
namespace emberfield::problem {
    namespace detail {
        /** @brief 仅发布完整热组装对象。@code auto p=ThermalAssemblyAccess::plan(p,a,o); @endcode */
        class ThermalAssemblyAccess {
        public:
            /** @brief 接管计划组成。@param p 上下文。@param a 符号图。@param o 选项。
             * @pre 均已验证。@return 只读计划。@code auto v=plan(p,a,o); @endcode */
            static std::shared_ptr<const ThermalAssemblyPlan> plan(std::shared_ptr<const PreparedThermalCase> p,
                                                                   std::shared_ptr<const assembly::AssemblyPlan> a,
                                                                   ThermalAssemblyOptions o) {
                auto v = std::shared_ptr<ThermalAssemblyPlan>(new ThermalAssemblyPlan);
                v->prepared_ = std::move(p);
                v->plan_ = std::move(a);
                v->options_ = o;
                return v;
            }
            /** @brief 接管数值组成。@param p 来源。@param s 空间及载荷。@param m 可选容量。
             * @pre 全部成功。@return 拥有模型。@code auto v=model(p,s,m); @endcode */
            static AssembledThermalModel model(std::shared_ptr<const ThermalAssemblyPlan> p, assembly::AssemblyValues s,
                                               std::optional<assembly::Matrix> m) {
                AssembledThermalModel v;
                v.plan_ = std::move(p);
                v.spatial_ = std::make_shared<const assembly::Matrix>(std::move(s.matrix_));
                v.load_ = std::make_shared<const Vector<double>>(std::move(s.load_));
                v.capacity_ = std::make_shared<const std::optional<assembly::Matrix>>(std::move(m));
                return v;
            }
        };
    } // namespace detail
    std::shared_ptr<const ThermalAssemblyPlan> buildThermalAssemblyPlan(std::shared_ptr<const PreparedThermalCase> p,
                                                                        ThermalAssemblyOptions o) {
        EMBERFIELD_NVTX_SCOPE_COLOR("emberfield::problem::buildThermalAssemblyPlan", emberfield::NvtxColor::Yellow);
        if (!p || !o.batchElements_ || !o.gpuLimits_.maxElements_ || !o.gpuLimits_.maxScalars_) {
            throw std::invalid_argument("组装上下文及批预算必须有效");
        }
        const auto cells = p->space()->cellCount(), sides = p->conditions()->naturalTerms().size();
        if (cells > o.limits_.maxStencils_ || sides > o.limits_.maxStencils_ - cells) {
            throw std::length_error("热模板数超限");
        }
        if (o.gpuLimits_.maxScalars_ < (p->space()->dimension() == 2 ? 18u : 48u)) {
            throw std::length_error("GPU预算不足一个体单元");
        }
        std::vector<assembly::DofList> stencils;
        stencils.reserve(cells + sides);
        for (std::size_t e = 0; e < cells; ++e) {
            const auto dofs = p->space()->cellDofs(e);
            stencils.emplace_back(dofs.begin(), dofs.end());
        }
        for (const auto& t : p->conditions()->naturalTerms()) {
            stencils.push_back(t.globalDofs());
        }
        auto graph = assembly::buildAssemblyPlan(p->space()->dofCount(), stencils, o.limits_);
        return detail::ThermalAssemblyAccess::plan(std::move(p), std::move(graph), o);
    }
    AssembledThermalModel assembleThermalModel(std::shared_ptr<const ThermalAssemblyPlan> plan) {
        EMBERFIELD_NVTX_SCOPE_COLOR("emberfield::problem::assembleThermalModel", emberfield::NvtxColor::Yellow);
        if (!plan) {
            throw std::invalid_argument("热组装计划为空");
        }
        const auto& p = *plan->prepared();
        if (hasTimeDependentFields(p.bound()->data().definition_)) {
            throw std::invalid_argument("时间表算例请使用ThermalEvolutionContext与assembleThermalState");
        }
        const auto& o = plan->options();
        const auto count = p.space()->cellCount();
        const auto cellBatch = std::min({o.batchElements_, o.gpuLimits_.maxElements_,
                                         o.gpuLimits_.maxScalars_ / (p.space()->dimension() == 2 ? 18u : 48u)});
        assembly::AssemblyAccumulator spatial(plan->plan());
        std::optional<assembly::AssemblyAccumulator> mass;
        if (p.bound()->data().definition_.analysis_.type_ == AnalysisType::Transient) {
            mass.emplace(plan->plan());
        }
        for (std::size_t first = 0; first < count;) {
            EMBERFIELD_NVTX_SCOPE("emberfield::problem::assembleThermalModel/cellBatch");
            const auto n = std::min(cellBatch, count - first);
            std::vector<std::size_t> rows(n);
            std::iota(rows.begin(), rows.end(), first);
            const auto batch = evaluateCellBatch(p, rows, o.gpuLimits_);
            spatial.addMatrix(first, n, batch.conductivity().values());
            spatial.addVector(first, n, batch.load().values());
            if (mass) {
                mass->addMatrix(first, n, batch.capacity()->values());
            }
            first += n;
        }
        const auto& terms = p.conditions()->naturalTerms();
        const auto& definitions = p.bound()->data().definition_.boundaries_;
        const auto sideBatch = std::min({o.batchElements_, o.gpuLimits_.maxElements_,
                                         o.gpuLimits_.maxScalars_ / (p.space()->dimension() == 2 ? 4u : 9u)});
        for (std::size_t first = 0; first < terms.size();) {
            EMBERFIELD_NVTX_SCOPE("emberfield::problem::assembleThermalModel/boundaryBatch");
            const auto type = definitions[terms[first].definitionIndex()].value_.index();
            std::size_t n = 1;
            while (n < std::min(sideBatch, terms.size() - first) &&
                   definitions[terms[first + n].definitionIndex()].value_.index() == type) {
                ++n;
            }
            std::vector<std::size_t> rows(n);
            std::iota(rows.begin(), rows.end(), first);
            const auto batch = evaluateBoundaryBatch(p, rows, o.gpuLimits_);
            if (const auto* f = std::get_if<thermal::GpuLocalVectorBatch>(&batch.contribution())) {
                spatial.addVector(count + first, n, f->values());
            }
            else {
                const auto& c = std::get<thermal::GpuConvectionBatch>(batch.contribution());
                spatial.addMatrix(count + first, n, c.matrix().values());
                spatial.addVector(count + first, n, c.load().values());
            }
            first += n;
        }
        std::optional<assembly::Matrix> capacity;
        if (mass) {
            capacity = std::move(*mass).release().matrix_;
        }
        return detail::ThermalAssemblyAccess::model(std::move(plan), std::move(spatial).release(), std::move(capacity));
    }
} // namespace emberfield::problem
