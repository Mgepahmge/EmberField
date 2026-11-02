/** @file thermal_assembly_test.cpp
 * @brief 全热模型GPU装配、分量锚定及完整稳态/瞬态系统验证。
 */
#include <Simulation/Case/System.cuh>
#include <Solver/Iterative/BiCgStab.cuh>
#include <Solver/Iterative/Cg.cuh>
#include "support/AssemblyFixture.cuh"
#include "support/CaseFixture.cuh"
namespace {
    using namespace thermofem;
    using namespace problem;
    using namespace assemblyfixture;
    /** @brief 准备独立测试模型。@param c 定义。@param data 网格。@pre 合法输入。
     * @return 完整准备。@code auto p=prepare(c); @endcode */
    std::shared_ptr<const PreparedThermalCase> prepare(CaseDefinition c,
                                                       mesh::MeshData data = casefixture::meshData()) {
        auto b = casefixture::bind(c, std::move(data));
        if (!b.bound_) {
            throw std::runtime_error("测试绑定失败");
        }
        auto s = buildTemperatureSpace(*b.bound_);
        if (!s.space_) {
            throw std::runtime_error("测试空间失败");
        }
        auto r = prepareThermalCase(b.bound_, s.space_);
        if (!r.ok()) {
            throw std::runtime_error("测试准备失败");
        }
        return r.prepared();
    }
    /** @brief 独立主机散加。@param global 行主序输出。@param n 全维度。
     * @param dofs 局部编号。@param values 局部方阵。@pre 形状相容。
     * @note 无返回。@code scatter(k,n,dofs,local.entries()); @endcode */
    void scatter(std::vector<double>& global, std::size_t n, std::span<const std::uint64_t> dofs,
                 const std::vector<double>& values) {
        for (std::size_t a = 0; a < dofs.size(); ++a) {
            for (std::size_t b = 0; b < dofs.size(); ++b) {
                global[dofs[a] * n + dofs[b]] += values[a * dofs.size() + b];
            }
        }
    }
    TEST_F(GpuAssembly, ThermalMixedTermsAndBatchSizesMatchIndependentDenseScatter) {
        auto c = casefixture::definition();
        casefixture::makeTransient(c);
        c.sources_ = {{"q1", casefixture::group(2, 11), ConstantScalar{100}},
                      {"q2", casefixture::group(2, 11), ConstantScalar{-5}}};
        c.naturalOverlap_ = NaturalOverlap::Add;
        c.boundaries_.push_back({"f", casefixture::group(1, 21), FluxBoundary{ConstantScalar{12}}});
        c.boundaries_.push_back(
            {"h1", casefixture::group(1, 22), ConvectionBoundary{ConstantScalar{3}, ConstantScalar{320}}});
        c.boundaries_.push_back(
            {"h2", casefixture::group(1, 22), ConvectionBoundary{ConstantScalar{5}, ConstantScalar{280}}});
        c.boundaries_.push_back({"top", casefixture::group(1, 23), FluxBoundary{ConstantScalar{-7}}});
        auto p = prepare(c);
        const auto n = p->space()->dofCount();
        std::vector<double> s(n * n, 0), m(n * n, 0), f(n, 0);
        for (std::size_t e = 0; e < p->space()->cellCount(); ++e) {
            auto t = evaluateCellTerms(*p, e);
            const auto dofs = p->space()->cellDofs(e);
            scatter(s, n, dofs, t.conductivity().entries());
            scatter(m, n, dofs, t.capacity()->entries());
            for (std::size_t a = 0; a < dofs.size(); ++a) {
                f[dofs[a]] += t.load()[a];
            }
        }
        for (std::size_t e = 0; e < p->conditions()->naturalTerms().size(); ++e) {
            const auto t = evaluateBoundaryTerm(*p, e);
            const auto& dofs = p->conditions()->naturalTerms()[e].globalDofs();
            if (const auto* flux = std::get_if<thermal::LocalVector>(&t)) {
                for (std::size_t a = 0; a < dofs.size(); ++a) {
                    f[dofs[a]] += (*flux)[a];
                }
            }
            else {
                const auto& h = std::get<thermal::ConvectionContribution>(t);
                scatter(s, n, dofs, h.matrix().entries());
                for (std::size_t a = 0; a < dofs.size(); ++a) {
                    f[dofs[a]] += h.load()[a];
                }
            }
        }
        for (std::size_t batch : {1, 2, 7}) {
            ThermalAssemblyOptions o;
            o.batchElements_ = batch;
            auto plan = buildThermalAssemblyPlan(p, o);
            auto model = assembleThermalModel(plan);
            near(dense(model.spatial()), s);
            near(dense(*model.capacity()), m);
            near(host(model.load().values()), f);
            auto again = assembleThermalModel(plan);
            EXPECT_EQ(host(again.spatial().values()), host(model.spatial().values()));
            EXPECT_EQ(host(again.load().values()), host(model.load().values()));
            EXPECT_EQ(host(model.spatial().columnIndices()), host(model.capacity()->columnIndices()));
            auto diagnostic = checkThermalSolvability(model);
            EXPECT_TRUE(diagnostic.canBuildSystem());
            EXPECT_TRUE(diagnostic.components_[0].hasCapacity_);
        }
    }
    TEST_F(GpuAssembly, SteadyLinearFieldSolvesWithCgAndBiCgStab) {
        auto c = casefixture::definition();
        c.boundaries_.push_back({"right", casefixture::group(1, 22), TemperatureBoundary{ConstantScalar{400}}});
        auto data = casefixture::meshData();
        data.nodes_.push_back({999, {0.5, 0.5, 0}, {}, {}});
        auto& body = data.elements_[0];
        body.tags_ = {100, 101, 102, 103};
        body.offsets_ = {0, 3, 6, 9, 12};
        body.nodes_ = {0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4};
        auto p = prepare(c, data);
        auto model = assembleThermalModel(buildThermalAssemblyPlan(p));
        auto system = buildThermalSteadySystem(model);
        std::vector<double> zero(5, 0);
        Vector<double> x{std::span<const double>(zero)};
        CgOptions<double> co;
        co.relativeTolerance_ = 1e-12;
        co.absoluteTolerance_ = 1e-11;
        Cg<double, assembly::Index> cg(co);
        cg.setup(system.matrix());
        auto r = cg.apply(system.rhs(), x);
        EXPECT_EQ(r.status_, SolveStatus::Converged);
        EXPECT_LT(r.residualNorm_, 1e-8);
        auto values = host(x.values());
        for (std::size_t d = 0; d < 5; ++d) {
            const auto node = *p->space()->dofInfo(d).node_;
            EXPECT_NEAR(values[d], 300 + 100 * data.nodes_[node].position_[0], 1e-8);
        }
        x.copyFromHost(zero);
        BiCgStabOptions<double> bo;
        bo.relativeTolerance_ = 1e-12;
        bo.absoluteTolerance_ = 1e-11;
        BiCgStab<double, assembly::Index> bicg(bo, std::make_unique<Identity<double, assembly::Index>>());
        bicg.setup(system.matrix());
        r = bicg.apply(system.rhs(), x);
        EXPECT_EQ(r.status_, SolveStatus::Converged);
        near(host(x.values()), values);
    }
    TEST_F(GpuAssembly, PureNeumannBalancedAndUnbalancedAreDistinguishedAndRejected) {
        auto c = casefixture::definition();
        c.boundaries_.clear();
        auto model = assembleThermalModel(buildThermalAssemblyPlan(prepare(c)));
        auto r = checkThermalSolvability(model);
        ASSERT_EQ(r.components_.size(), 1);
        EXPECT_FALSE(r.canBuildSystem());
        EXPECT_TRUE(r.components_[0].loadCompatible_);
        EXPECT_THROW((void)buildThermalSteadySystem(model), std::domain_error);
        c.sources_ = {{"q", casefixture::group(2, 11), ConstantScalar{10}}};
        auto other = assembleThermalModel(buildThermalAssemblyPlan(prepare(c)));
        r = checkThermalSolvability(other);
        EXPECT_FALSE(r.components_[0].loadCompatible_);
        EXPECT_NEAR(r.components_[0].netLoad_, 0.1, 1e-14);
        EXPECT_THROW((void)buildThermalSteadySystem(other), std::domain_error);
        EXPECT_THROW((void)checkThermalSolvability(other, -1), std::invalid_argument);
    }
    TEST_F(GpuAssembly, OnlyPositiveConvectionAnchorsSteadyComponent) {
        auto c = casefixture::definition();
        c.boundaries_.clear();
        c.boundaries_.push_back(
            {"h", casefixture::group(1, 22), ConvectionBoundary{ConstantScalar{0}, ConstantScalar{300}}});
        auto zero = assembleThermalModel(buildThermalAssemblyPlan(prepare(c)));
        EXPECT_FALSE(checkThermalSolvability(zero).canBuildSystem());
        std::get<ConvectionBoundary>(c.boundaries_[0].value_).coefficient_ = ConstantScalar{3};
        auto positive = assembleThermalModel(buildThermalAssemblyPlan(prepare(c)));
        EXPECT_TRUE(checkThermalSolvability(positive).components_[0].hasPositiveConvection_);
        EXPECT_NO_THROW((void)buildThermalSteadySystem(positive));
    }
    TEST_F(GpuAssembly, EveryDisconnectedComponentMustHaveItsOwnAnchor) {
        auto data = casefixture::meshData();
        const auto oldNodes = data.nodes_;
        for (auto node : oldNodes) {
            node.tag_ += 10000000000ULL;
            node.position_[0] += 3;
            data.nodes_.push_back(node);
        }
        auto extra = data.elements_[0];
        extra.tags_ = {700, 701};
        for (auto& node : extra.nodes_) {
            node += 4;
        }
        data.elements_.push_back(extra);
        auto c = casefixture::definition();
        auto p = prepare(c, data);
        auto model = assembleThermalModel(buildThermalAssemblyPlan(p));
        auto r = checkThermalSolvability(model);
        ASSERT_EQ(r.components_.size(), 2);
        EXPECT_TRUE(r.components_[0].hasDirichlet_);
        EXPECT_FALSE(r.components_[1].anchored());
        EXPECT_FALSE(r.canBuildSystem());
        EXPECT_THROW((void)buildThermalSteadySystem(model), std::domain_error);
    }
    TEST_F(GpuAssembly, TransientPureNeumannAndNonzeroDirichletPreserveConstantField) {
        for (bool constrained : {false, true}) {
            auto c = casefixture::definition();
            casefixture::makeTransient(c);
            if (!constrained) {
                c.boundaries_.clear();
            }
            auto p = prepare(c);
            auto model = assembleThermalModel(buildThermalAssemblyPlan(p));
            EXPECT_TRUE(checkThermalSolvability(model).canBuildSystem());
            const auto initial = *p->conditions()->initialTemperature();
            Vector<double> previous{std::span<const double>(initial)};
            for (double dt : {0.1, 1., 0.017}) {
                auto s = buildThermalStepSystem(model, previous, dt);
                CgOptions<double> o;
                o.absoluteTolerance_ = 1e-7;
                Cg<double, assembly::Index> cg(o);
                cg.setup(s.matrix());
                Vector<double> x = previous;
                auto r = cg.apply(s.rhs(), x);
                EXPECT_EQ(r.status_, SolveStatus::Converged);
                near(host(x.values()), initial);
            }
            EXPECT_THROW((void)buildThermalSteadySystem(model), std::invalid_argument);
            std::vector<double> invalid(initial.size(), -1);
            previous.copyFromHost(invalid);
            EXPECT_THROW((void)buildThermalStepSystem(model, previous, 0.1), std::invalid_argument);
        }
    }
    TEST(ThermalTime, ClipsFinalIntervalAndRejectsInvalidOrNonprogressingState) {
        TimeOptions o{0, 1, 0.3, 4};
        double current = 0;
        std::size_t count = 0;
        while (auto s = nextTimeStep(o, current, count)) {
            ++count;
            EXPECT_EQ(s->index_, count);
            EXPECT_EQ(s->start_, current);
            EXPECT_GT(s->duration_, 0);
            current = s->end_;
            if (count == 4) {
                EXPECT_NEAR(s->duration_, 0.1, 1e-15);
            }
        }
        EXPECT_EQ(count, 4);
        EXPECT_EQ(current, 1);
        TimeOptions decimal{0, 10, 0.1, 100};
        double time = 0;
        std::size_t steps = 0;
        while (auto s = nextTimeStep(decimal, time, steps)) {
            time = s->end_;
            ++steps;
        }
        EXPECT_EQ(steps, 100);
        EXPECT_EQ(time, 10);
        EXPECT_THROW((void)nextTimeStep(o, 0.9, 4), std::length_error);
        EXPECT_THROW((void)nextTimeStep(o, -1, 0), std::invalid_argument);
        EXPECT_THROW((void)nextTimeStep(o, 0, 5), std::invalid_argument);
        o = {1e20, 1e20 + 1e6, 1e-10, 10};
        EXPECT_THROW((void)nextTimeStep(o, o.start_, 0), std::overflow_error);
    }
    TEST_F(GpuAssembly, ThermalPlanBudgetsAndModelTypeAreValidated) {
        auto p = prepare(casefixture::definition());
        ThermalAssemblyOptions o;
        EXPECT_THROW((void)buildThermalAssemblyPlan({}), std::invalid_argument);
        EXPECT_THROW((void)assembleThermalModel({}), std::invalid_argument);
        o.batchElements_ = 0;
        EXPECT_THROW((void)buildThermalAssemblyPlan(p, o), std::invalid_argument);
        o = {};
        o.limits_.maxStencils_ = 1;
        EXPECT_THROW((void)buildThermalAssemblyPlan(p, o), std::length_error);
        o = {};
        o.gpuLimits_.maxScalars_ = 17;
        EXPECT_THROW((void)buildThermalAssemblyPlan(p, o), std::length_error);
        auto m = assembleThermalModel(buildThermalAssemblyPlan(p));
        Vector<double> previous(4);
        EXPECT_THROW((void)buildThermalStepSystem(m, previous, 1), std::invalid_argument);
    }
} // namespace
