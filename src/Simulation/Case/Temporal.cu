/** @file Temporal.cu @brief 时间快照、事件调度及按依赖更新的GPU热组装。 */
#include <Numerics/LinearAlgebra/Operations.cuh>
#include <Simulation/Case/Temporal.cuh>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include "ThermalSampling.cuh"

namespace emberfield::problem {
    namespace {
        /** @brief 收集单场区间内节点。@param field 场。@param time 区间。@param events 输出。
         * @pre 已验证。@note 无返回。@code appendEvents(f,t,e); @endcode */
        void appendEvents(const ScalarField& field, const TimeOptions& time, std::vector<double>& events) {
            if (const auto* table = std::get_if<TimeTableScalar>(&field)) {
                for (const auto& p : table->points_) {
                    if (p[0] >= time.start_ && p[0] <= time.end_) {
                        events.push_back(p[0]);
                    }
                }
            }
        }
        /** @brief 检查运行设备。@param expected 设备。@pre CUDA可用。@note 无返回，错误异常。
         * @code requireDevice(id); @endcode */
        void requireDevice(int expected) {
            int device = -1;
            const auto error = cudaGetDevice(&device);
            if (error != cudaSuccess) {
                throw CudaError(error, "temporal device");
            }
            if (device != expected) {
                throw std::invalid_argument("演化上下文当前CUDA设备错配");
            }
        }
        /** @brief 从起点计算名义节点。@param t 时间设置。@param index 独立名义编号。
         * @pre index>0。@return 有限节点。@code auto e=nominalPoint(t,i); @endcode */
        double nominalPoint(const TimeOptions& t, std::size_t index) {
            const long double value = static_cast<long double>(t.start_) + static_cast<long double>(index) * t.step_;
            double result = static_cast<double>(std::min(value, static_cast<long double>(t.end_)));
            if (!std::isfinite(result)) {
                throw std::overflow_error("名义时间节点不可表示");
            }
            const double snap =
                8 * std::numeric_limits<double>::epsilon() * std::max(std::abs(t.end_), std::abs(t.start_));
            if (result > t.start_ && result < t.end_ && t.end_ - result <= snap) {
                result = t.end_;
            }
            return result;
        }
        /** @brief 按现有GPU预算限制批大小。@param p 计划。@param scalars 最大局部标量数。
         * @pre 正预算。@return 正批大小。@code auto n=batchSize(p,48); @endcode */
        std::size_t batchSize(const ThermalAssemblyPlan& p, std::size_t scalars) {
            const auto& o = p.options();
            const auto n = std::min({o.batchElements_, o.gpuLimits_.maxElements_, o.gpuLimits_.maxScalars_ / scalars});
            if (!n) {
                throw std::length_error("时间积分GPU预算不足单个实体");
            }
            return n;
        }
        /** @brief GPU计算一类完整体源向量。@param plan 计划。@param state 数值参数。
         * @pre 同源。@return 设备载荷。@code auto f=volumeLoad(p,s); @endcode */
        Vector<double> volumeLoad(const ThermalAssemblyPlan& plan, const EvaluatedThermalState& state) {
            const auto& p = *plan.prepared();
            const auto limits = plan.options().gpuLimits_;
            assembly::AssemblyAccumulator result(plan.plan());
            const auto maxBatch = batchSize(plan, p.space()->dimension() == 2 ? 18 : 48);
            for (std::size_t first = 0; first < p.space()->cellCount();) {
                const auto n = std::min(maxBatch, p.space()->cellCount() - first);
                std::vector<fem::CellIntegrationData> tables;
                for (std::size_t i = 0; i < n; ++i) {
                    tables.push_back(fem::evaluateCellIntegration(p.geometry(), first + i, 2));
                }
                auto batch = thermal::uploadCellBatch(tables, limits);
                std::vector<double> samples(n * batch.pointCount(), 0), scales(n, detail::thermalSectionScale(p));
                auto sum = thermal::integrateVolumeSourceBatch(batch, samples, scales);
                std::size_t slots = 0;
                const auto& c = *p.conditions();
                for (std::size_t i = 0; i < n; ++i) {
                    slots = std::max(slots, c.sourceOffsets()[first + i + 1] - c.sourceOffsets()[first + i]);
                }
                for (std::size_t slot = 0; slot < slots; ++slot) {
                    std::fill(samples.begin(), samples.end(), 0);
                    for (std::size_t i = 0; i < n; ++i) {
                        const auto begin = c.sourceOffsets()[first + i], end = c.sourceOffsets()[first + i + 1];
                        if (slot < end - begin) {
                            std::fill_n(samples.begin() + i * batch.pointCount(), batch.pointCount(),
                                        state.sources()[c.sourceIndices()[begin + slot]]);
                        }
                    }
                    auto contribution = thermal::integrateVolumeSourceBatch(batch, samples, scales);
                    sum = thermal::addLocalVectorBatches(batch, sum, contribution);
                }
                result.addVector(first, n, sum.values());
                first += n;
            }
            return std::move(result).release().load_;
        }
        /** @brief GPU计算通量或对流类，保持自然项原顺序。@param plan 计划。@param state 参数。
         * @param flux 是否通量。@param matrix 是否组装H。@pre 同源。@return 当前类矩阵载荷。
         * @code auto h=boundaryValues(p,s,false,true); @endcode */
        assembly::AssemblyValues boundaryValues(const ThermalAssemblyPlan& plan, const EvaluatedThermalState& state,
                                                bool flux, bool matrix) {
            const auto& p = *plan.prepared();
            const auto& terms = p.conditions()->naturalTerms();
            const auto& boundaries = p.bound()->data().definition_.boundaries_;
            assembly::AssemblyAccumulator result(plan.plan());
            const auto maxBatch = batchSize(plan, p.space()->dimension() == 2 ? 4 : 9);
            for (std::size_t first = 0; first < terms.size();) {
                const bool isFlux =
                    std::holds_alternative<FluxBoundary>(boundaries[terms[first].definitionIndex()].value_);
                std::size_t n = 1;
                while (n < std::min(maxBatch, terms.size() - first) &&
                       std::holds_alternative<FluxBoundary>(boundaries[terms[first + n].definitionIndex()].value_) ==
                           isFlux) {
                    ++n;
                }
                if (isFlux == flux) {
                    std::vector<std::size_t> rows(n);
                    std::iota(rows.begin(), rows.end(), first);
                    auto batch = evaluateBoundaryBatch(p, rows, plan.options().gpuLimits_, &state);
                    if (flux) {
                        result.addVector(p.space()->cellCount() + first, n,
                                         std::get<thermal::GpuLocalVectorBatch>(batch.contribution()).values());
                    }
                    else {
                        const auto& contribution = std::get<thermal::GpuConvectionBatch>(batch.contribution());
                        if (matrix) {
                            result.addMatrix(p.space()->cellCount() + first, n, contribution.matrix().values());
                        }
                        result.addVector(p.space()->cellCount() + first, n, contribution.load().values());
                    }
                }
                first += n;
            }
            return std::move(result).release();
        }
    } // namespace
    std::shared_ptr<const ThermalEvolutionPlan> buildThermalEvolutionPlan(std::shared_ptr<const PreparedThermalCase> p,
                                                                          ThermalAssemblyOptions options) {
        if (!p || !p->bound()->data().definition_.time_) {
            throw std::invalid_argument("演化计划需要已准备瞬态算例");
        }
        auto result = std::shared_ptr<ThermalEvolutionPlan>(new ThermalEvolutionPlan);
        result->assemblyPlan_ = buildThermalAssemblyPlan(p, options);
        const auto& def = p->bound()->data().definition_;
        const auto& time = *def.time_;
        for (const auto& source : def.sources_) {
            appendEvents(source.value_, time, result->events_);
        }
        for (const auto& boundary : def.boundaries_) {
            if (const auto* g = std::get_if<TemperatureBoundary>(&boundary.value_)) {
                appendEvents(g->temperature_, time, result->events_);
            }
            else if (const auto* q = std::get_if<FluxBoundary>(&boundary.value_)) {
                appendEvents(q->flux_, time, result->events_);
            }
            else {
                const auto& h = std::get<ConvectionBoundary>(boundary.value_);
                appendEvents(h.coefficient_, time, result->events_);
                appendEvents(h.ambientTemperature_, time, result->events_);
            }
        }
        std::sort(result->events_.begin(), result->events_.end());
        result->events_.erase(std::unique(result->events_.begin(), result->events_.end()), result->events_.end());
        return result;
    }
    EvaluatedThermalState evaluateThermalState(std::shared_ptr<const ThermalEvolutionPlan> plan, double time) {
        if (!plan || !std::isfinite(time)) {
            throw std::invalid_argument("时间快照参数非法");
        }
        const auto& def = plan->assemblyPlan()->prepared()->bound()->data().definition_;
        if (time < def.time_->start_ || time > def.time_->end_) {
            throw std::invalid_argument("快照时刻不在算例区间");
        }
        EvaluatedThermalState result;
        result.plan_ = std::move(plan);
        result.time_ = time;
        for (const auto& source : def.sources_) {
            result.sources_.push_back(evaluate(source.value_, time));
        }
        result.primary_.resize(def.boundaries_.size());
        result.ambient_.resize(def.boundaries_.size());
        std::vector<double> prescribed(def.boundaries_.size());
        for (std::size_t i = 0; i < def.boundaries_.size(); ++i) {
            const auto& boundary = def.boundaries_[i];
            if (const auto* g = std::get_if<TemperatureBoundary>(&boundary.value_)) {
                prescribed[i] = evaluate(g->temperature_, time);
            }
            else if (const auto* q = std::get_if<FluxBoundary>(&boundary.value_)) {
                result.primary_[i] = evaluate(q->flux_, time);
            }
            else {
                const auto& h = std::get<ConvectionBoundary>(boundary.value_);
                result.primary_[i] = evaluate(h.coefficient_, time);
                result.ambient_[i] = evaluate(h.ambientTemperature_, time);
                if (!std::isfinite(result.primary_[i] * result.ambient_[i])) {
                    throw std::overflow_error("对流h*T∞溢出: " + boundary.id_);
                }
            }
        }
        const auto& c = result.plan_->assemblyPlan()->prepared()->conditions()->constraints();
        for (std::size_t i = 0; i < c.dofs().size(); ++i) {
            const auto begin = c.sourceOffsets()[i], end = c.sourceOffsets()[i + 1];
            const double value = prescribed.at(c.sourceIndices().at(begin));
            for (auto j = begin; j < end; ++j) {
                if (prescribed[c.sourceIndices()[j]] != value) {
                    throw std::domain_error("时变规定温度冲突 DOF " + std::to_string(c.dofs()[i]));
                }
            }
            result.constraints_.push_back(value);
        }
        return result;
    }
    ThermalTimeCursor::ThermalTimeCursor(std::shared_ptr<const ThermalEvolutionPlan> plan) : plan_(std::move(plan)) {
        if (!plan_) {
            throw std::invalid_argument("时间游标计划为空");
        }
        const auto& t = *plan_->assemblyPlan()->prepared()->bound()->data().definition_.time_;
        current_ = t.start_;
        if (nominalPoint(t, t.maxSteps_) < t.end_) {
            throw std::length_error("名义时间步超过maxSteps");
        }
        event_ = std::upper_bound(plan_->events().begin(), plan_->events().end(), current_) - plan_->events().begin();
        // 有界预演，只计数不展开时间数组；实际游标保持起点。
        auto probe = *this;
        while (auto step = probe.peek()) {
            probe.accept(*step);
        }
        planned_ = probe.completed_;
    }
    std::optional<TimeStep> ThermalTimeCursor::peek() const {
        const auto& t = *plan_->assemblyPlan()->prepared()->bound()->data().definition_.time_;
        if (current_ == t.end_) {
            return {};
        }
        if (completed_ >= t.maxSteps_) {
            throw std::length_error("事件插入后的时间步超过maxSteps");
        }
        const double nominal = nominalPoint(t, nominal_);
        const double event = event_ < plan_->events().size() ? plan_->events()[event_] : t.end_;
        const double end = std::min(nominal, event);
        if (!std::isfinite(end - current_) || end <= current_) {
            throw std::overflow_error("事件时间步无法严格推进");
        }
        return TimeStep{completed_ + 1, current_, end, end - current_};
    }
    void ThermalTimeCursor::accept(const TimeStep& step) {
        auto expected = peek();
        if (!expected || expected->index_ != step.index_ || expected->start_ != step.start_ ||
            expected->end_ != step.end_ || expected->duration_ != step.duration_) {
            throw std::invalid_argument("时间游标候选错配");
        }
        const auto& t = *plan_->assemblyPlan()->prepared()->bound()->data().definition_.time_;
        current_ = step.end_;
        ++completed_;
        if (current_ == nominalPoint(t, nominal_)) {
            if (current_ != t.end_) {
                if (nominal_ == SIZE_MAX) {
                    throw std::overflow_error("名义时间编号溢出");
                }
                ++nominal_;
            }
        }
        while (event_ < plan_->events().size() && plan_->events()[event_] == current_) {
            ++event_;
        }
    }
    /** @brief 独占运行缓存，所有返回资源只读共享。 */
    struct ThermalEvolutionContext::Impl {
        std::shared_ptr<const ThermalEvolutionPlan> plan_; ///< 来源。
        int device_{}; ///< 固定设备。
        std::shared_ptr<const assembly::Matrix> k_, spatial_; ///< 固定K与当前S。
        std::shared_ptr<const std::optional<assembly::Matrix>> mass_; ///< 固定M。
        std::shared_ptr<const Vector<double>> volume_, flux_, ambient_; ///< 三类载荷。
        std::shared_ptr<const Vector<double>> load_; ///< 当前总载荷。
        std::shared_ptr<const EvaluatedThermalState> last_; ///< 上次数值依赖。
        std::size_t revision_{}; ///< 当前S版本。
    };
    ThermalEvolutionContext::ThermalEvolutionContext(std::shared_ptr<const ThermalEvolutionPlan> plan) :
        impl_(std::make_unique<Impl>()) {
        if (!plan) {
            throw std::invalid_argument("演化上下文计划为空");
        }
        impl_->plan_ = std::move(plan);
        const auto error = cudaGetDevice(&impl_->device_);
        if (error != cudaSuccess) {
            throw CudaError(error, "temporal context");
        }
        const auto& planRef = *impl_->plan_->assemblyPlan();
        const auto& p = *planRef.prepared();
        const auto initial = evaluateThermalState(impl_->plan_, p.bound()->data().definition_.time_->start_);
        assembly::AssemblyAccumulator k(planRef.plan()), mass(planRef.plan());
        const auto maxBatch = batchSize(planRef, p.space()->dimension() == 2 ? 18 : 48);
        for (std::size_t first = 0; first < p.space()->cellCount();) {
            const auto n = std::min(maxBatch, p.space()->cellCount() - first);
            std::vector<std::size_t> rows(n);
            std::iota(rows.begin(), rows.end(), first);
            auto batch = evaluateCellBatch(p, rows, planRef.options().gpuLimits_, &initial);
            k.addMatrix(first, n, batch.conductivity().values());
            mass.addMatrix(first, n, batch.capacity()->values());
            first += n;
        }
        impl_->k_ = std::make_shared<const assembly::Matrix>(std::move(k).release().matrix_);
        impl_->mass_ = std::make_shared<const std::optional<assembly::Matrix>>(std::move(mass).release().matrix_);
    }
    ThermalEvolutionContext::~ThermalEvolutionContext() = default;
    ThermalEvolutionContext::ThermalEvolutionContext(ThermalEvolutionContext&&) noexcept = default;
    ThermalEvolutionContext& ThermalEvolutionContext::operator=(ThermalEvolutionContext&&) noexcept = default;
    ThermalNumericState ThermalEvolutionContext::assemble(const EvaluatedThermalState& state) {
        if (!impl_ || !state.plan() || state.plan() != impl_->plan_) {
            throw std::invalid_argument("时间组装来源错配");
        }
        auto& c = *impl_;
        requireDevice(c.device_);
        const auto& plan = *c.plan_->assemblyPlan();
        const auto& def = plan.prepared()->bound()->data().definition_;
        bool fluxChanged = !c.last_, hChanged = !c.last_, ambientChanged = !c.last_;
        if (c.last_) {
            for (std::size_t i = 0; i < def.boundaries_.size(); ++i) {
                if (std::holds_alternative<FluxBoundary>(def.boundaries_[i].value_)) {
                    fluxChanged |= state.primary()[i] != c.last_->primary()[i];
                }
                if (std::holds_alternative<ConvectionBoundary>(def.boundaries_[i].value_)) {
                    hChanged |= state.primary()[i] != c.last_->primary()[i];
                    ambientChanged |= state.ambient()[i] != c.last_->ambient()[i];
                }
            }
        }
        auto volume = c.volume_, flux = c.flux_, ambient = c.ambient_;
        auto spatial = c.spatial_;
        auto revision = c.revision_;
        const bool sourceChanged = !c.last_ || state.sources() != c.last_->sources();
        if (sourceChanged) {
            volume = std::make_shared<const Vector<double>>(volumeLoad(plan, state));
        }
        if (fluxChanged) {
            flux = std::make_shared<const Vector<double>>(boundaryValues(plan, state, true, false).load_);
        }
        if (hChanged || ambientChanged) {
            auto values = boundaryValues(plan, state, false, hChanged);
            ambient = std::make_shared<const Vector<double>>(std::move(values.load_));
            if (hChanged) {
                if (revision == SIZE_MAX) {
                    throw std::overflow_error("矩阵版本溢出");
                }
                spatial = std::make_shared<const assembly::Matrix>(linalg::add(*c.k_, values.matrix_));
                ++revision;
            }
        }
        auto load = c.load_;
        if (sourceChanged || fluxChanged || hChanged || ambientChanged) {
            load = std::make_shared<const Vector<double>>(linalg::add(linalg::add(*volume, *flux), *ambient));
            systems::validateSystemValues(*spatial, *load);
        }
        auto model = std::shared_ptr<AssembledThermalModel>(new AssembledThermalModel);
        model->plan_ = c.plan_->assemblyPlan();
        model->spatial_ = spatial;
        model->capacity_ = c.mass_;
        model->load_ = load;
        ThermalNumericState result;
        result.parameters_ = std::make_shared<const EvaluatedThermalState>(state);
        model->parameters_ = result.parameters_;
        result.model_ = std::move(model);
        result.matrixRevision_ = revision;
        c.volume_ = std::move(volume);
        c.flux_ = std::move(flux);
        c.ambient_ = std::move(ambient);
        c.spatial_ = std::move(spatial);
        c.load_ = std::move(load);
        c.revision_ = revision;
        c.last_ = result.parameters_;
        return result;
    }
    systems::LinearSystem buildThermalStepSystem(const ThermalNumericState& numeric, const Vector<double>& previous,
                                                 const EvaluatedThermalState& old, const TimeStep& step) {
        const auto& state = numeric.parameters();
        const auto& model = numeric.model();
        if (state.plan() != old.plan() || old.time() != step.start_ || state.time() != step.end_ ||
            !std::isfinite(step.duration_) || step.duration_ <= 0 || step.duration_ != step.end_ - step.start_ ||
            model.plan() != state.plan()->assemblyPlan()) {
            throw std::invalid_argument("时变系统来源或时间错配");
        }
        const auto& dofs = model.plan()->prepared()->conditions()->constraints().dofs();
        systems::validateTemperatureState(previous, dofs, old.constraints());
        return systems::applyDirichlet(systems::buildBackwardEulerSystem(model.spatial(), *model.capacity(),
                                                                         model.load(), previous, step.duration_),
                                       dofs, state.constraints());
    }
} // namespace emberfield::problem
