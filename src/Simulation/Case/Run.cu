/** @file Run.cu @brief 正式稳态/瞬态运行编排，不在求解器内部执行文件操作。 */
#include <IO/Case/Case.cuh>
#include <IO/Mesh/Msh.cuh>
#include <IO/Result/Archive.cuh>
#include <NVTXProfiler.cuh>
#include <Numerics/LinearAlgebra/IterationContext.cuh>
#include <Simulation/Case/Run.cuh>
#include <Simulation/Case/System.cuh>
#include <Simulation/Case/Temporal.cuh>
#include <Solver/Iterative/BiCgStab.cuh>
#include <Solver/Iterative/Cg.cuh>
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include "../../IO/Case/Console.cuh"
#include "StepSystemCache.cuh"
namespace emberfield::problem {
    namespace {
        using Clock = std::chrono::steady_clock;
        /** @brief 完成阶段墙钟秒数。@param start 起点。@pre 单调时钟。@return 秒数。@code auto t=elapsed(start);
         * @endcode */
        double elapsed(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }
        /** @brief 输出带严重性和定位信息的输入诊断。
         * @param observer 借用呈现端。@param d 诊断集合。@pre 无。@note 无返回值，文本不作为机器协议。
         * @code diagnostics(d); @endcode */
        void diagnostics(const Diagnostics& d, RunObserver& observer) {
            std::ostringstream text;
            for (const auto& x : d.entries()) {
                text << (x.severity_ == Severity::Error ? "[错误] " : "[警告] ") << x.code_ << ": " << x.message_;
                if (!x.path_.empty()) {
                    text << " (" << x.path_ << ')';
                }
                if (!x.source_.empty()) {
                    text << " @ " << x.source_;
                    if (x.line_) {
                        text << ':' << *x.line_;
                        if (x.column_) {
                            text << ':' << *x.column_;
                        }
                    }
                }
                text << '\n';
                if (!x.suggestion_.empty()) {
                    text << "         建议: " << x.suggestion_ << '\n';
                }
            }
            if (d.omitted() != 0) {
                text << "[提示] 还有 " << d.omitted() << " 条诊断未显示，请调整诊断预算\n";
            }
            observer.onEvent({.kind_ = RunEventKind::Message, .message_ = text.str(), .error_ = true});
        }
        /** @brief 输出成功准备的算例与矩阵规模。
         * @param observer 借用呈现端。@param input 配置路径。@param binding 活动网格与定义。@param model 已组装模型。
         * @pre binding和model同源。@note 无返回值。
         * @code printCaseSummary(input,bound.data(),model); @endcode */
        void printCaseSummary(const std::filesystem::path& input, const BindingData& binding,
                              const AssembledThermalModel& model, RunObserver& observer) {
            std::ostringstream text;
            const auto& c = binding.definition_;
            text << "[算例] " << (c.name_.empty() ? input.stem().string() : c.name_) << '\n'
                 << "  分析         " << (c.analysis_.type_ == AnalysisType::Steady ? "稳态" : "瞬态") << " / "
                 << (c.analysis_.space_ == SpaceModel::Planar2d ? "二维平面" : "三维实体") << '\n'
                 << "  网格         " << c.mesh_.path_.string() << '\n'
                 << "  规模         " << binding.mesh_->data().nodes_.size() << " 节点 / " << binding.active_.size()
                 << " 活动单元 / " << model.spatial().rowCount() << " DOF / " << model.spatial().nonZeroCount()
                 << " nnz\n"
                 << "  求解器       " << (c.linearSolver_.method_ == LinearMethod::Cg ? "CG" : "BiCGStab")
                 << " + Identity, rtol=" << c.linearSolver_.relativeTolerance_
                 << ", atol=" << c.linearSolver_.absoluteTolerance_ << '\n';
            if (c.time_) {
                text << "  时间         [" << c.time_->start_ << ", " << c.time_->end_ << "] s, dt=" << c.time_->step_
                     << " s (后向 Euler)\n";
            }
            text << "\n[进度] 温度范围单位 K；完整逐步记录见 steps.csv\n" << std::flush;
            observer.onEvent({.kind_ = RunEventKind::Message, .message_ = text.str()});
        }
        /** @brief 展示已发布状态的简短进度，不替代结构化结果。
         * @param observer 借用呈现端。@param record 状态记录。@param minimumK 最低温度。@param maximumK 最高温度。
         * @pre 数值有限且状态已发布。@note 无返回值。
         * @code printProgress(record,minK,maxK); @endcode */
        void printProgress(const io::StepRecord& record, double minimumK, double maximumK, RunObserver& observer) {
            std::ostringstream text;
            std::ostringstream line;
            line << "  [" << std::setw(6) << record.step_ << "]";
            if (record.time_) {
                line << " t=" << std::scientific << std::setprecision(3) << *record.time_ << " s";
            }
            if (record.solve_) {
                line << "  iter=" << record.solve_->iterations_;
            }
            if (record.acceptedResidual_) {
                line << "  |r|=" << std::scientific << std::setprecision(3) << *record.acceptedResidual_;
            }
            line << "  T=[" << std::fixed << std::setprecision(3) << minimumK << ", " << maximumK << "]";
            if (record.kind_ == "initial") {
                line << "  初态";
            }
            text << line.str() << '\n' << std::flush;
            observer.onEvent({.kind_ = RunEventKind::Message, .message_ = text.str()});
        }
        /** @brief 创建运行时求解器。@param o 已验证配置。@pre 迭代数可表示int。@return 独占求解器。@code auto
         * s=solver(o); @endcode */
        std::unique_ptr<Solver<double, assembly::Index>> solver(const LinearOptions& o) {
            EMBERFIELD_NVTX_SCOPE_COLOR("emberfield::problem::solver", emberfield::NvtxColor::Yellow);
            if (o.maxIterations_ > INT_MAX) {
                throw std::length_error("迭代上限超出int");
            }
            if (o.method_ == LinearMethod::Cg) {
                CgOptions<double> p;
                p.maxIterations_ = static_cast<int>(o.maxIterations_);
                p.relativeTolerance_ = o.relativeTolerance_;
                p.absoluteTolerance_ = o.absoluteTolerance_;
                return std::make_unique<Cg<double, assembly::Index>>(p);
            }
            BiCgStabOptions<double> p;
            p.maxIterations_ = static_cast<int>(o.maxIterations_);
            p.relativeTolerance_ = o.relativeTolerance_;
            p.absoluteTolerance_ = o.absoluteTolerance_;
            return std::make_unique<BiCgStab<double, assembly::Index>>(
                p, std::make_unique<Identity<double, assembly::Index>>());
        }
        /** @brief 数值状态的稳定字符串。@param status 求解终止状态。@pre 枚举有效。@return 机器字符串。@code auto
         * s=statusName(status); @endcode */
        std::string statusName(SolveStatus status) {
            switch (status) {
            case SolveStatus::Converged:
                return "converged";
            case SolveStatus::MaxIterations:
                return "maxIterations";
            case SolveStatus::Breakdown:
                return "breakdown";
            case SolveStatus::NonFinite:
                return "nonFinite";
            }
            return "error";
        }
    } // namespace
    /** @brief 同步数值运行内部实现，外层负责资源销毁后的Finished通知。
     * @param input 算例。@param output 输出目录。@param interrupted 可空信号标志。
     * @param limits 预算。@param observer 借用呈现端。@param control 可空原子取消器。
     * @pre 同公开运行入口。@return 同公开退出码。@code auto result=runCaseImpl(in,out,nullptr,limits,sink,nullptr);
     * @endcode */
    static int runCaseImpl(const std::filesystem::path& input, const std::filesystem::path& output,
                           const volatile std::sig_atomic_t* interrupted, const ResourceLimits& limits,
                           RunObserver& observer, RunControl* control) {
        const auto cancelled = [&] { return (interrupted && *interrupted) || (control && control->cancelled()); };
        const auto checkpoint = [&] {
            if (cancelled()) {
                throw std::runtime_error("用户请求中断");
            }
        };
        const auto phase = [&](RunStage stage) {
            checkpoint();
            observer.onEvent({.kind_ = RunEventKind::Stage, .stage_ = stage});
        };
        const auto message = [&](const std::string& text, bool error = false) {
            observer.onEvent({.kind_ = RunEventKind::Message, .message_ = text, .error_ = error});
        };
        EMBERFIELD_NVTX_SCOPE("emberfield::problem::runCase");
        const auto started = Clock::now();
        std::unique_ptr<io::ResultArchive> archive;
        bool outputStage = false, solveCompleted = false;
        try {
            phase(RunStage::Loading);
            message("[准备] 读取算例与网格\n");
            auto stage = Clock::now();
            auto loaded = io::loadCase(input, limits.case_, limits.output_);
            diagnostics(loaded.diagnostics_, observer);
            if (!loaded.definition_) {
                return 2;
            }
            loaded.definition_->output_.directory_ = std::filesystem::absolute(output).lexically_normal();
            auto msh = io::loadMsh(loaded.definition_->mesh_.path_, limits.mesh_);
            const double inputS = elapsed(stage);
            phase(RunStage::Preparing);
            message("[准备] 绑定区域并建立温度自由度\n");
            stage = Clock::now();
            auto binding =
                bindCase(*loaded.definition_, std::make_shared<const mesh::Mesh>(std::move(msh.mesh_)), limits.case_);
            diagnostics(binding.diagnostics_, observer);
            if (!binding.bound_) {
                return 2;
            }
            auto space = buildTemperatureSpace(*binding.bound_, limits.space_);
            diagnostics(space.diagnostics_, observer);
            if (!space.space_) {
                return 2;
            }
            auto prepared = prepareThermalCase(binding.bound_, space.space_, limits.preparation_);
            diagnostics(prepared.diagnostics(), observer);
            if (!prepared.ok()) {
                return 2;
            }
            ThermalPostprocessPlan postprocess(prepared.prepared());
            const double preparationS = elapsed(stage);
            phase(RunStage::Assembling);
            message("[准备] 计算单元贡献并组装稀疏矩阵\n");
            stage = Clock::now();
            const auto& c = binding.bound_->data().definition_;
            std::shared_ptr<const ThermalEvolutionPlan> evolutionPlan;
            std::optional<ThermalEvolutionContext> evolution;
            std::optional<ThermalTimeCursor> cursor;
            std::optional<EvaluatedThermalState> oldState;
            std::optional<ThermalNumericState> numeric;
            if (c.time_) {
                evolutionPlan = buildThermalEvolutionPlan(prepared.prepared(), limits.assembly_);
                try {
                    cursor.emplace(evolutionPlan);
                }
                catch (const std::exception& e) {
                    {
                        std::ostringstream text;
                        text << "[错误] TIME_STEP_BUDGET: " << e.what() << '\n';
                        message(text.str(), true);
                    }
                    return 2;
                }
            }
            auto model = [&] {
                EMBERFIELD_NVTX_SCOPE("emberfield::problem::runCase/initialAssembly");
                if (!c.time_) {
                    return assembleThermalModel(buildThermalAssemblyPlan(prepared.prepared(), limits.assembly_),
                                                &observer);
                }
                oldState = evaluateThermalState(evolutionPlan, c.time_->start_);
                evolution.emplace(evolutionPlan, &observer);
                numeric = assembleThermalState(*evolution, *oldState);
                return numeric->model();
            }();
            double assemblyS = elapsed(stage);
            phase(RunStage::Saving);
            message("[准备] 创建结果目录\n");
            stage = Clock::now();
            outputStage = true;
            archive = std::make_unique<io::ResultArchive>(output, input, model, postprocess);
            if (cursor) {
                const auto eventCount =
                    std::count_if(evolutionPlan->events().begin(), evolutionPlan->events().end(),
                                  [&](double t) { return t > c.time_->start_ && t < c.time_->end_; });
                archive->timeEvaluation(static_cast<std::size_t>(eventCount), cursor->plannedSteps());
            }
            for (const auto& d : loaded.diagnostics_.entries()) {
                archive->addDiagnostic(d, "input");
            }
            for (const auto& d : binding.diagnostics_.entries()) {
                archive->addDiagnostic(d, "binding");
            }
            for (const auto& d : space.diagnostics_.entries()) {
                archive->addDiagnostic(d, "space");
            }
            for (const auto& d : prepared.diagnostics().entries()) {
                archive->addDiagnostic(d, "preparation");
            }
            outputStage = false;
            archive->timing("input", inputS);
            archive->timing("preparation", preparationS);
            archive->timing("assembly", assemblyS);
            double outputS = elapsed(stage), setupS = 0, solveS = 0, postprocessS = 0;
            const auto& constraints = prepared.prepared()->conditions()->constraints();
            printCaseSummary(input, binding.bound_->data(), model, observer);
            const auto meshPath = c.mesh_.path_.string();
            const auto caseName = c.name_.empty() ? input.stem().string() : c.name_;
            CaseSummary summary{caseName,
                                meshPath,
                                c.time_ ? "Transient / Backward Euler" : "Steady",
                                c.linearSolver_.method_ == LinearMethod::Cg ? "CG + Identity" : "BiCGStab + Identity",
                                binding.bound_->data().mesh_->data().nodes_.size(),
                                binding.bound_->data().active_.size(),
                                model.spatial().rowCount(),
                                model.spatial().nonZeroCount(),
                                c.linearSolver_.relativeTolerance_,
                                c.linearSolver_.absoluteTolerance_};
            const auto totalSteps = cursor ? cursor->plannedSteps() : 1;
            observer.onEvent({.kind_ = RunEventKind::Metadata, .summary_ = summary, .totalSteps_ = totalSteps});
            const auto stride = cursor ? std::max<std::size_t>(1, (cursor->plannedSteps() + 23) / 24) : 1;
            if (cursor && hasTimeDependentFields(c)) {
                std::size_t linear = 0, previous = 0;
                const auto count = [&](const ScalarField& f) {
                    if (const auto* table = std::get_if<TimeTableScalar>(&f)) {
                        if (table->interpolation_ == TimeInterpolation::Linear) {
                            ++linear;
                        }
                        else {
                            ++previous;
                        }
                    }
                };
                for (const auto& source : c.sources_) {
                    count(source.value_);
                }
                for (const auto& boundary : c.boundaries_) {
                    if (const auto* g = std::get_if<TemperatureBoundary>(&boundary.value_)) {
                        count(g->temperature_);
                    }
                    else if (const auto* q = std::get_if<FluxBoundary>(&boundary.value_)) {
                        count(q->flux_);
                    }
                    else {
                        const auto& h = std::get<ConvectionBoundary>(boundary.value_);
                        count(h.coefficient_);
                        count(h.ambientTemperature_);
                    }
                }
                {
                    std::ostringstream text;
                    text << "  时变条件     " << linear << " 线性 / " << previous << " 分段保持\n"
                         << "  时间采样     步末右连续 / 时间表节点对齐 / " << cursor->plannedSteps() << " 实际步\n";
                    message(text.str());
                }
                if (previous) {
                    {
                        std::ostringstream text;
                        text << "  提示         跳变处仍存在时间离散误差；事件对齐不等于精确积分\n";
                        message(text.str());
                    }
                }
            }
            std::size_t completedSteps = 0;
            auto iterative = solver(c.linearSolver_);
            IterationSubscription<double, assembly::Index> subscription(*iterative, &observer);
            std::optional<std::size_t> setupRevision;
            std::optional<double> setupDt;
            detail::ThermalStepSystemCache stepSystems;
            std::unique_ptr<linalg::IterationContext<double, assembly::Index>> acceptance;
            // 候选经规定值投影和真实残差复核后才接受；历史状态保留至后处理完成。
            auto advance = [&](const systems::LinearSystem& system, const Vector<double>* previous,
                               io::StepRecord record, bool final) -> std::optional<Vector<double>> {
                EMBERFIELD_NVTX_SCOPE("emberfield::problem::runCase/advance");
                if (cancelled()) {
                    throw std::runtime_error("用户请求中断");
                }
                try {
                    phase(RunStage::SettingUp);
                    auto tick = Clock::now();
                    const auto revision = numeric ? numeric->matrixRevision() : 0;
                    if (!setupRevision || *setupRevision != revision || setupDt != record.dt_) {
                        EMBERFIELD_NVTX_SCOPE("emberfield::problem::runCase/solverSetup");
                        auto nextAcceptance =
                            std::make_unique<linalg::IterationContext<double, assembly::Index>>(system.matrix());
                        iterative->setup(system.matrix());
                        acceptance = std::move(nextAcceptance);
                        setupRevision = revision;
                        setupDt = record.dt_;
                    }
                    Vector<double> candidate(system.rhs().size());
                    if (previous && c.linearSolver_.initialGuess_ == InitialGuess::PreviousStep) {
                        candidate.copyDataFrom(*previous);
                    }
                    else {
                        auto code = cudaMemset(candidate.deviceData(), 0, candidate.size() * sizeof(double));
                        if (code != cudaSuccess) {
                            throw CudaError(code, "initial guess");
                        }
                    }
                    record.setupS_ = elapsed(tick);
                    setupS += record.setupS_;
                    tick = Clock::now();
                    phase(RunStage::Solving);
                    record.solve_ = iterative->apply(system.rhs(), candidate);
                    record.solveS_ = elapsed(tick);
                    solveS += record.solveS_;
                    record.status_ = statusName(record.solve_->status_);
                    if (record.solve_->status_ != SolveStatus::Converged) {
                        record.kind_ = "failedAttempt";
                        record.errorCode_ = "SOLVER_FAILED";
                        outputStage = true;
                        archive->publishStep(record, nullptr, false);
                        archive->finish(false, record.errorCode_, "线性求解未收敛");
                        outputStage = false;
                        {
                            std::ostringstream text;
                            text << "[错误] 第 " << record.step_ << " 步线性求解未收敛 (" << record.status_ << ")\n";
                            message(text.str(), true);
                        }
                        return {};
                    }
                    phase(RunStage::Checking);
                    candidate = systems::projectTemperatureState(candidate, constraints.dofs(),
                                                                 numeric ? numeric->parameters().constraints()
                                                                         : constraints.temperatures());
                    tick = Clock::now();
                    auto& check = *acceptance;
                    Vector<double> residual(candidate.size());
                    check.residual(system.rhs(), candidate, residual);
                    record.acceptedResidual_ = check.norm(residual);
                    const auto norm = check.norm(system.rhs());
                    setupS += elapsed(tick);
                    const double tolerance =
                        std::max(c.linearSolver_.absoluteTolerance_, c.linearSolver_.relativeTolerance_ * norm);
                    if (!std::isfinite(*record.acceptedResidual_) || *record.acceptedResidual_ > tolerance) {
                        record.kind_ = "failedAttempt";
                        record.status_ = "error";
                        record.errorCode_ = "ACCEPTANCE_RESIDUAL";
                        outputStage = true;
                        archive->publishStep(record, nullptr, false);
                        archive->finish(false, record.errorCode_, "规定温度投影后真实残差未达标");
                        outputStage = false;
                        {
                            std::ostringstream text;
                            text << "[错误] 第 " << record.step_ << " 步接受残差未达标\n";
                            message(text.str(), true);
                        }
                        return {};
                    }
                    observer.onEvent({.kind_ = RunEventKind::SolveAccepted,
                                      .step_ = record.step_,
                                      .acceptedResidual_ = record.acceptedResidual_});
                    archive->accepted(record.step_, record.time_);
                    phase(RunStage::Postprocessing);
                    tick = Clock::now();
                    const bool saveFrame = io::shouldSave(c.output_, c.time_.has_value(), record.step_, final, false);
                    auto fields = numeric ? postprocess.evaluate(*numeric, candidate, previous, &*oldState, record.dt_,
                                                                 record.step_, false, saveFrame)
                                          : postprocess.evaluate(model, candidate, previous, record.dt_, record.step_,
                                                                 record.time_, false, saveFrame);
                    record.postprocessS_ = elapsed(tick);
                    postprocessS += record.postprocessS_;
                    outputStage = true;
                    tick = Clock::now();
                    phase(RunStage::Saving);
                    archive->publishStep(record, &fields, saveFrame);
                    outputS += elapsed(tick);
                    outputStage = false;
                    if (!c.time_ || record.step_ == 1 || final || record.step_ % stride == 0) {
                        printProgress(record, fields.minimumK_, fields.maximumK_, observer);
                    }
                    return candidate;
                }
                catch (...) {
                    if (!outputStage) {
                        record.kind_ = "failedAttempt";
                        record.status_ = "error";
                        record.errorCode_ = "RUN_ERROR";
                        try {
                            outputStage = true;
                            archive->publishStep(record, nullptr, false);
                            outputStage = false;
                        }
                        catch (const std::exception& failure) {
                            {
                                std::ostringstream text;
                                text << "[错误] 记录失败尝试失败: " << failure.what() << '\n';
                                message(text.str(), true);
                            }
                        }
                    }
                    throw;
                }
            };
            if (!c.time_) {
                io::StepRecord record;
                observer.onEvent({.kind_ = RunEventKind::StepStarted, .totalSteps_ = totalSteps});
                phase(RunStage::BuildingSystem);
                if (!advance(buildThermalSteadySystem(model), nullptr, record, true)) {
                    return 3;
                }
                observer.onEvent({.kind_ = RunEventKind::StepCommitted, .step_ = 1, .totalSteps_ = totalSteps});
            }
            else {
                Vector<double> previous(
                    std::span<const double>(*prepared.prepared()->conditions()->initialTemperature()));
                double time = c.time_->start_;
                archive->accepted(0, time);
                io::StepRecord record;
                record.kind_ = "initial";
                record.status_ = "provided";
                record.time_ = time;
                stage = Clock::now();
                const bool saveInitial = io::shouldSave(c.output_, true, 0, false, true);
                auto fields = postprocess.evaluate(*numeric, previous, nullptr, nullptr, {}, 0, true, saveInitial);
                record.postprocessS_ = elapsed(stage);
                postprocessS += record.postprocessS_;
                outputStage = true;
                stage = Clock::now();
                archive->publishStep(record, &fields, saveInitial);
                outputS += elapsed(stage);
                outputStage = false;
                printProgress(record, fields.minimumK_, fields.maximumK_, observer);
                while (auto step = cursor->peek()) {
                    EMBERFIELD_NVTX_SCOPE("emberfield::problem::runCase/timeStep");
                    if (cancelled()) {
                        throw std::runtime_error("用户请求中断");
                    }
                    record = io::StepRecord{};
                    record.step_ = step->index_;
                    record.time_ = step->end_;
                    record.dt_ = step->duration_;
                    observer.onEvent({.kind_ = RunEventKind::StepStarted,
                                      .step_ = step->index_,
                                      .totalSteps_ = totalSteps,
                                      .time_ = record.time_,
                                      .dt_ = record.dt_});
                    phase(RunStage::Assembling);
                    std::optional<EvaluatedThermalState> parameters;
                    std::optional<systems::LinearSystem> system;
                    try {
                        stage = Clock::now();
                        parameters = evaluateThermalState(evolutionPlan, step->end_);
                        numeric = assembleThermalState(*evolution, *parameters);
                        model = numeric->model();
                        phase(RunStage::BuildingSystem);
                        system = stepSystems.build(*numeric, previous, *oldState, *step);
                        assemblyS += elapsed(stage);
                        archive->timing("assembly", assemblyS);
                    }
                    catch (...) {
                        record.kind_ = "failedAttempt";
                        record.status_ = "error";
                        record.errorCode_ = "TIME_ASSEMBLY_ERROR";
                        outputStage = true;
                        archive->publishStep(record, nullptr, false);
                        outputStage = false;
                        throw;
                    }
                    auto next = advance(*system, &previous, record, step->end_ == c.time_->end_);
                    if (!next) {
                        return 3;
                    }
                    previous = std::move(*next);
                    time = step->end_;
                    oldState = std::move(*parameters);
                    cursor->accept(*step);
                    completedSteps = step->index_;
                    observer.onEvent({.kind_ = RunEventKind::StepCommitted,
                                      .step_ = completedSteps,
                                      .totalSteps_ = totalSteps,
                                      .time_ = record.time_,
                                      .dt_ = record.dt_});
                }
            }
            phase(RunStage::Finalizing);
            solveCompleted = true;
            archive->timing("solverSetup", setupS);
            archive->timing("solve", solveS);
            archive->timing("postprocess", postprocessS);
            archive->timing("output", outputS);
            archive->timing("total", elapsed(started));
            outputStage = true;
            archive->finish(true);
            outputStage = false;
            {
                std::ostringstream text;
                text << "\n[完成] " << (c.time_ ? "瞬态" : "稳态");
                message(text.str());
            }
            if (c.time_) {
                {
                    std::ostringstream text;
                    text << "，共 " << completedSteps << " 步";
                    message(text.str());
                }
            }
            {
                std::ostringstream text;
                text << "\n  用时         " << std::fixed << std::setprecision(3) << elapsed(started) << " s\n"
                     << "  结果         " << output.string() << "\n"
                     << "  摘要         " << (output / "run-summary.json").string() << '\n';
                message(text.str());
            }
            return 0;
        }
        catch (const std::exception& e) {
            {
                std::ostringstream text;
                text << "[错误] " << e.what() << '\n';
                message(text.str(), true);
            }
            if (archive) {
                try {
                    archive->timing("total", elapsed(started));
                    const bool wasCancelled = cancelled();
                    archive->finish(solveCompleted,
                                    outputStage        ? "OUTPUT_IO"
                                        : wasCancelled ? "USER_INTERRUPTED"
                                                       : "RUN_ERROR",
                                    e.what(), outputStage, wasCancelled);
                }
                catch (const std::exception& nested) {
                    {
                        std::ostringstream text;
                        text << "[错误] 终结摘要失败: " << nested.what() << '\n';
                        message(text.str(), true);
                    }
                }
            }
            return cancelled() ? 130 : 1;
        }
    }
    int runCase(const std::filesystem::path& input, const std::filesystem::path& output,
                const volatile std::sig_atomic_t* interrupted, const ResourceLimits& limits) {
        return runCase(input, output, interrupted, limits, nullptr, nullptr);
    }
    int runCase(const std::filesystem::path& input, const std::filesystem::path& output,
                const volatile std::sig_atomic_t* interrupted, const ResourceLimits& limits, RunObserver* observer,
                RunControl* control) {
        cli::ConsolePresenter console;
        auto& sink = observer ? *observer : static_cast<RunObserver&>(console);
        const auto started = Clock::now();
        const int result = runCaseImpl(input, output, interrupted, limits, sink, control);
        sink.onEvent({.kind_ = RunEventKind::Finished,
                      .stage_ = result == 0 ? RunStage::Completed
                          : result == 130   ? RunStage::Cancelled
                                            : RunStage::Failed,
                      .elapsedS_ = elapsed(started),
                      .exitCode_ = result});
        return result;
    }
} // namespace emberfield::problem
