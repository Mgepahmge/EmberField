/** @file solve-case.cu
 * @brief 加载完整JSON与MSH，GPU组装稳态或多步后向Euler系统并调用已有求解器。
 */
#include <IO/Case/Case.cuh>
#include <IO/Mesh/Msh.cuh>
#include <Simulation/Case/System.cuh>
#include <Solver/Iterative/BiCgStab.cuh>
#include <Solver/Iterative/Cg.cuh>
#include <iostream>
#include <limits>
namespace {
    using namespace thermofem;
    /** @brief 输出输入诊断。@param d 诊断。@pre 无。@note 无返回值。
     * @code diagnostics(r.diagnostics()); @endcode */
    void diagnostics(const problem::Diagnostics& d) {
        for (const auto& e : d.entries()) {
            std::cerr << e.code_ << ": " << e.message_ << '\n';
        }
    }
    /** @brief 按算例选择方法和恒等预条件子。@param o 已验证配置。
     * @pre maxIterations可表示为int。@return 独占求解器。
     * @code auto solver=makeSolver(options); @endcode */
    std::unique_ptr<Solver<double, assembly::Index>> makeSolver(const problem::LinearOptions& o) {
        if (o.maxIterations_ > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::length_error("求解迭代上限超出int");
        }
        if (o.method_ == problem::LinearMethod::Cg) {
            CgOptions<double> options;
            options.maxIterations_ = static_cast<int>(o.maxIterations_);
            options.relativeTolerance_ = o.relativeTolerance_;
            options.absoluteTolerance_ = o.absoluteTolerance_;
            return std::make_unique<Cg<double, assembly::Index>>(options);
        }
        BiCgStabOptions<double> options;
        options.maxIterations_ = static_cast<int>(o.maxIterations_);
        options.relativeTolerance_ = o.relativeTolerance_;
        options.absoluteTolerance_ = o.absoluteTolerance_;
        return std::make_unique<BiCgStab<double, assembly::Index>>(
            options, std::make_unique<Identity<double, assembly::Index>>());
    }
    /** @brief 求解并要求真实残差达标。@param s 系统。@param o 配置。
     * @param physical 可选历史场，只用于previousStep初猜。@pre 尺寸相容。
     * @return 温度解；失败runtime_error。@code auto x=solve(s,o,previous); @endcode */
    Vector<double> solve(const systems::LinearSystem& s, const problem::LinearOptions& o,
                         const Vector<double>* physical) {
        auto solver = makeSolver(o);
        solver->setup(s.matrix());
        Vector<double> x(s.rhs().size());
        if (physical && o.initialGuess_ == problem::InitialGuess::PreviousStep) {
            x.copyDataFrom(*physical);
        }
        else {
            const auto code = cudaMemset(x.deviceData(), 0, x.size() * sizeof(double));
            if (code != cudaSuccess) {
                throw CudaError(code, "cudaMemset initial guess");
            }
        }
        const auto report = solver->apply(s.rhs(), x);
        std::cout << "iterations=" << report.iterations_ << " residual=" << report.residualNorm_ << '\n';
        if (report.status_ != SolveStatus::Converged) {
            throw std::runtime_error("线性求解未收敛");
        }
        return x;
    }
} // namespace
/** @brief 运行完整算例到A,b及温度解。@param argc 参数数。@param argv 配置路径。
 * @pre 有可用GPU。@return 0成功，1异常，2输入失败。
 * @code ./solve-case example/cases/transient.json @endcode */
int main(int argc, char** argv) {
    using namespace thermofem;
    if (argc != 2) {
        std::cerr << "用法: solve-case <case.json>\n";
        return 2;
    }
    try {
        auto input = io::loadCase(argv[1]);
        diagnostics(input.diagnostics_);
        if (!input.definition_) {
            return 2;
        }
        auto msh = io::loadMsh(input.definition_->mesh_.path_);
        auto binding = problem::bindCase(*input.definition_, std::make_shared<const mesh::Mesh>(std::move(msh.mesh_)));
        diagnostics(binding.diagnostics_);
        if (!binding.bound_) {
            return 2;
        }
        auto space = problem::buildTemperatureSpace(*binding.bound_);
        diagnostics(space.diagnostics_);
        if (!space.space_) {
            return 2;
        }
        auto prepared = problem::prepareThermalCase(binding.bound_, space.space_);
        diagnostics(prepared.diagnostics());
        if (!prepared.ok()) {
            return 2;
        }
        auto model = problem::assembleThermalModel(problem::buildThermalAssemblyPlan(prepared.prepared()));
        std::cout << "DOF=" << model.spatial().rowCount() << " nnz=" << model.spatial().nonZeroCount() << '\n';
        const auto& def = binding.bound_->data().definition_;
        Vector<double> result;
        if (def.analysis_.type_ == problem::AnalysisType::Steady) {
            result = solve(problem::buildThermalSteadySystem(model), def.linearSolver_, nullptr);
        }
        else {
            result = Vector<double>(std::span<const double>(*prepared.prepared()->conditions()->initialTemperature()));
            const auto& c = prepared.prepared()->conditions()->constraints();
            double time = def.time_->start_;
            std::size_t completed = 0;
            while (auto step = problem::nextTimeStep(*def.time_, time, completed)) {
                auto system = problem::buildThermalStepSystem(model, result, step->duration_);
                auto candidate = solve(system, def.linearSolver_, &result);
                result = systems::projectTemperatureState(candidate, c.dofs(), c.temperatures());
                time = step->end_;
                completed = step->index_;
                std::cout << "step=" << completed << " time=" << time << " dt=" << step->duration_ << '\n';
            }
        }
        std::vector<double> temperatures(result.size());
        result.copyToHost(temperatures);
        const auto [minimum, maximum] = std::minmax_element(temperatures.begin(), temperatures.end());
        std::cout << "temperature[K]=[" << *minimum << ", " << *maximum << "]\n";
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
