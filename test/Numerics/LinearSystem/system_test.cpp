/** @file system_test.cpp
 * @brief 以独立稠密公式验证后向Euler、同时规定值消元与数值范围检查。
 */
#include <Numerics/LinearSystem/Operations.cuh>
#include <limits>
#include "support/AssemblyFixture.cuh"
namespace {
    using namespace thermofem;
    using namespace assemblyfixture;
    using namespace systems;
    TEST_F(GpuAssembly, BackwardEulerMatchesDenseHistoryBeforeNonzeroElimination) {
        auto s = matrix(3, {4, -1, -2, -1, 5, -1, -2, -1, 6});
        auto m = matrix(3, {2, 1, 0.5, 1, 3, 1, 0.5, 1, 4});
        const std::vector<double> f{2, -3, 4}, t{300, 310, 320};
        Vector<double> load{std::span<const double>(f)}, previous{std::span<const double>(t)};
        for (double dt : {0.1, 2., 17.}) {
            auto system = buildBackwardEulerSystem(s, m, load, previous, dt);
            auto ref = dense(s), mass = dense(m);
            std::vector<double> rhs = f;
            for (std::size_t i = 0; i < 3; ++i) {
                for (std::size_t j = 0; j < 3; ++j) {
                    ref[3 * i + j] += mass[3 * i + j] / dt;
                    rhs[i] += mass[3 * i + j] / dt * t[j];
                }
            }
            near(dense(system.matrix()), ref);
            near(host(system.rhs().values()), rhs);
            const std::vector<std::uint64_t> ids{0, 2};
            const std::vector<double> g{300, 320};
            auto final = applyDirichlet(system, ids, g);
            auto before = ref;
            rhs[1] -= ref[3] * 300 + ref[5] * 320;
            ref = {1, 0, 0, 0, ref[4], 0, 0, 0, 1};
            rhs[0] = 300;
            rhs[2] = 320;
            near(dense(final.matrix()), ref);
            near(host(final.rhs().values()), rhs);
            EXPECT_EQ(dense(system.matrix()), before);
            EXPECT_EQ(host(final.matrix().rowOffsets()), host(system.matrix().rowOffsets()));
            EXPECT_EQ(host(final.matrix().columnIndices()), host(system.matrix().columnIndices()));
        }
        EXPECT_EQ(host(load.values()), f);
        EXPECT_EQ(host(previous.values()), t);
        EXPECT_EQ(dense(s), (std::vector<double>{4, -1, -2, -1, 5, -1, -2, -1, 6}));
    }
    TEST_F(GpuAssembly, EmptyAllAndMultipleConstraintsPreserveFullGraph) {
        auto a = matrix(2, {2, -1, -1, 2});
        std::vector<double> fv{1, 2};
        Vector<double> f{std::span<const double>(fv)};
        auto raw = buildSteadySystem(a, f);
        auto same = applyDirichlet(raw, {}, {});
        EXPECT_EQ(dense(same.matrix()), dense(a));
        std::vector<std::uint64_t> ids{0, 1};
        std::vector<double> g{5, -3};
        auto all = applyDirichlet(raw, ids, g);
        EXPECT_EQ(dense(all.matrix()), (std::vector<double>{1, 0, 0, 1}));
        EXPECT_EQ(host(all.rhs().values()), g);
        EXPECT_EQ(all.matrix().nonZeroCount(), 4);
        EXPECT_THROW((void)applyDirichlet(raw, ids, std::span<const double>(g).first(1)), std::invalid_argument);
        ids = {1, 0};
        EXPECT_THROW((void)applyDirichlet(raw, ids, g), std::invalid_argument);
        ids = {0, 0};
        EXPECT_THROW((void)applyDirichlet(raw, ids, g), std::invalid_argument);
        ids = {0, 2};
        EXPECT_THROW((void)applyDirichlet(raw, ids, g), std::invalid_argument);
        ids = {0, 1};
        g[1] = std::numeric_limits<double>::quiet_NaN();
        EXPECT_THROW((void)applyDirichlet(raw, ids, g), std::invalid_argument);
    }
    TEST_F(GpuAssembly, RejectsNoncanonicalGraphsAndNonfiniteInputs) {
        using I = assembly::Index;
        const std::vector<I> r{0, 2, 4}, c{1, 0, 0, 1}, duplicate{0, 0, 0, 1}, missing{1, 1};
        const std::vector<double> v{2, -1, -1, 2}, b{1, 1};
        Vector<double> f{std::span<const double>(b)};
        Matrix unsorted(2, 2, r, c, v), duplicates(2, 2, r, duplicate, v);
        EXPECT_THROW((void)buildSteadySystem(unsorted, f), std::invalid_argument);
        EXPECT_THROW((void)buildSteadySystem(duplicates, f), std::invalid_argument);
        Matrix noDiagonal(2, 2, std::vector<I>{0, 1, 2}, missing, b);
        EXPECT_THROW((void)buildSteadySystem(noDiagonal, f), std::invalid_argument);
        auto good = matrix(2, v), other = matrix(3, std::vector<double>(9, 1));
        EXPECT_THROW((void)buildBackwardEulerSystem(good, other, f, f, 1), std::invalid_argument);
        for (double dt : {0., -1., std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
            EXPECT_THROW((void)buildBackwardEulerSystem(good, good, f, f, dt), std::invalid_argument);
        }
        std::vector<double> bad{std::numeric_limits<double>::infinity(), 1};
        Vector<double> nonfinite{std::span<const double>(bad)};
        EXPECT_THROW((void)buildSteadySystem(good, nonfinite), std::invalid_argument);
        Vector<double> shortVector(1);
        EXPECT_THROW((void)buildSteadySystem(good, shortVector), std::invalid_argument);
        bad = {1, std::numeric_limits<double>::quiet_NaN()};
        auto invalid = matrix(1, {bad[1]});
        Vector<double> one(1);
        EXPECT_THROW((void)buildSteadySystem(invalid, one), std::invalid_argument);
    }
    TEST_F(GpuAssembly, RejectsPatternMismatchWithSameNonzeroCount) {
        using I = assembly::Index;
        Matrix a(3, 3, std::vector<I>{0, 2, 4, 5}, std::vector<I>{0, 1, 0, 1, 2}, std::vector<double>(5, 1));
        Matrix b(3, 3, std::vector<I>{0, 1, 3, 5}, std::vector<I>{0, 1, 2, 1, 2}, std::vector<double>(5, 1));
        Vector<double> v{std::span<const double>(std::vector<double>{1, 1, 1})};
        EXPECT_THROW((void)buildBackwardEulerSystem(a, b, v, v, 1), std::invalid_argument);
    }
    TEST_F(GpuAssembly, ReportsScalingHistoryAndEliminationOverflow) {
        const double maximum = std::numeric_limits<double>::max();
        auto a = matrix(1, {maximum}), zero = matrix(1, {0});
        const std::vector<double> one{1}, two{2}, zeros{0};
        Vector<double> f{std::span<const double>(one)}, t{std::span<const double>(two)};
        EXPECT_THROW((void)buildBackwardEulerSystem(a, a, f, t, 0.5), std::overflow_error);
        EXPECT_THROW((void)buildBackwardEulerSystem(zero, a, f, t, 1), std::overflow_error);
        auto tiny = matrix(1, {std::numeric_limits<double>::denorm_min()});
        EXPECT_THROW((void)buildBackwardEulerSystem(zero, tiny, f, t, maximum), std::overflow_error);
        auto cross = matrix(2, {1, maximum, maximum, 1});
        Vector<double> rhs{std::span<const double>(std::vector<double>{0, 0})};
        auto raw = buildSteadySystem(cross, rhs);
        EXPECT_THROW((void)applyDirichlet(raw, std::vector<std::uint64_t>{1}, two), std::overflow_error);
        EXPECT_EQ(dense(raw.matrix()), dense(cross));
    }
    TEST_F(GpuAssembly, TemperatureStateValidationIsIndependent) {
        std::vector<double> data{300, 301};
        Vector<double> t{std::span<const double>(data)};
        std::vector<std::uint64_t> d{0};
        std::vector<double> g{300};
        EXPECT_NO_THROW(validateTemperatureState(t, d, g));
        auto projected = projectTemperatureState(t, d, std::vector<double>{310});
        EXPECT_EQ(host(projected.values()), (std::vector<double>{310, 301}));
        EXPECT_EQ(host(t.values()), data);
        EXPECT_THROW((void)projectTemperatureState(t, d, std::vector<double>{-1}), std::invalid_argument);
        g[0] = 301;
        EXPECT_THROW(validateTemperatureState(t, d, g), std::invalid_argument);
        data[1] = -1;
        t.copyFromHost(data);
        EXPECT_THROW(validateTemperatureState(t, {}, {}), std::invalid_argument);
        data[1] = std::numeric_limits<double>::infinity();
        t.copyFromHost(data);
        EXPECT_THROW(validateTemperatureState(t, {}, {}), std::invalid_argument);
    }
} // namespace
