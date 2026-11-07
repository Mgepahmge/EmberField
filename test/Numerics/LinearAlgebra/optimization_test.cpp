/** @file optimization_test.cpp @brief 验证只读快照、规范化快路径和设备后处理的所有权与数值契约。 */
#include <Numerics/LinearAlgebra/IterationContext.cuh>
#include <Numerics/LinearAlgebra/SpmvPlan.cuh>
#include <Numerics/LinearSystem/StepPlan.cuh>
#include <Simulation/Thermal/PostprocessOperations.cuh>
#include <gtest/gtest.h>
#include <limits>
#include "support/AssemblyFixture.cuh"
namespace {
    using namespace emberfield;
    /** @brief CUDA正确性夹具，无设备时明确跳过。@code OptimizationGpu fixture; @endcode */
    class OptimizationGpu : public ::testing::Test {
    protected:
        /** @brief 选择设备，无参数无前提。@return 无。@code SetUp(); @endcode */
        void SetUp() override {
            int n{};
            if (cudaGetDeviceCount(&n) != cudaSuccess || !n) {
                GTEST_SKIP();
            }
            ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        }
    };
    TEST_F(OptimizationGpu, SnapshotSurvivesRetainedWritablePointerAndSourceDestruction) {
        const std::vector<int> rp{0, 2, 3}, ci{0, 2, 1};
        const std::vector<double> v{2, 3, 4};
        std::unique_ptr<linalg::detail::SpmvPlan<double, int>> plan;
        {
            CsrMatrix<double, int> a(2, 3, rp, ci, v);
            double* retained = a.deviceValues();
            auto frozen = linalg::detail::freezeCanonical(a);
            plan = std::make_unique<linalg::detail::SpmvPlan<double, int>>(frozen);
            const double changed = 999;
            ASSERT_EQ(cudaMemcpy(retained, &changed, sizeof(double), cudaMemcpyHostToDevice), cudaSuccess);
            auto shared = frozen;
            EXPECT_EQ(shared.matrix().deviceValues(), frozen.matrix().deviceValues());
            linalg::IterationContext<double, int> context(shared);
            EXPECT_EQ(context.matrix().deviceValues(), frozen.matrix().deviceValues());
        }
        Vector<double> x(std::vector<double>{1, 2, 3}), y(2);
        plan->multiply(x, y);
        EXPECT_EQ(linalg::detail::hostCopy(y), (std::vector<double>{11, 8}));
        Vector<double> second(std::vector<double>{2, 3, 4}), out(2);
        plan->multiply(second, out);
        EXPECT_EQ(linalg::detail::hostCopy(out), (std::vector<double>{16, 12}));
        EXPECT_THROW(plan->multiply(y, y), std::invalid_argument);
    }
    TEST_F(OptimizationGpu, DuplicateFallbackWideIndexAndMutableFiniteChecks) {
        const std::vector<std::int64_t> rp{0, 3, 4}, ci{1, 0, 1, 1};
        CsrMatrix<double, std::int64_t> a(2, 2, rp, ci, std::vector<double>{1, 2, 3, 0});
        auto frozen = linalg::detail::freezeCanonical(a);
        EXPECT_EQ(linalg::detail::hostCopy(frozen.matrix().columnIndices()), (std::vector<std::int64_t>{0, 1, 1}));
        EXPECT_EQ(linalg::detail::hostCopy(frozen.matrix().values()), (std::vector<double>{2, 4, 0}));
        a.copyValuesFromHost(std::vector<double>{1, 2, std::numeric_limits<double>::infinity(), 0});
        EXPECT_THROW(linalg::detail::freezeCanonical(a), std::invalid_argument);
        CsrMatrix<double, int> canonical(1, 1, std::vector<int>{0, 1}, std::vector<int>{0}, std::vector<double>{1});
        auto first = linalg::detail::freezeCanonical(canonical);
        canonical.copyValuesFromHost(std::vector<double>{std::numeric_limits<double>::quiet_NaN()});
        EXPECT_THROW(linalg::detail::freezeCanonical(canonical), std::invalid_argument);
        EXPECT_EQ(linalg::detail::hostCopy(first.matrix().values()), (std::vector<double>{1}));
    }
    TEST_F(OptimizationGpu, CanonicalAcceptsOtherSignedIndexTypes) {
        CsrMatrix<double, long long> input(1, 2, std::vector<long long>{0, 3}, std::vector<long long>{1, 0, 1},
                                           std::vector<double>{2, 3, 4});
        auto canonical = linalg::detail::canonical<long long>(input);
        EXPECT_EQ(linalg::detail::hostCopy(canonical.columnIndices()), (std::vector<long long>{0, 1}));
        EXPECT_EQ(linalg::detail::hostCopy(canonical.values()), (std::vector<double>{3, 6}));
        auto second = linalg::detail::canonical<long long>(canonical);
        EXPECT_EQ(linalg::detail::hostCopy(second.values()), (std::vector<double>{3, 6}));
    }
    TEST_F(OptimizationGpu, EmptyAndZeroNonzeros) {
        CsrMatrix<double, int> empty;
        linalg::detail::SpmvPlan<double, int> plan(linalg::detail::freezeCanonical(empty));
        Vector<double> x, y;
        EXPECT_NO_THROW(plan.multiply(x, y));
        CsrMatrix<float, int> zero(2, 3, std::vector<int>{0, 0, 0}, {}, {});
        linalg::detail::SpmvPlan<float, int> zeroPlan(linalg::detail::freezeCanonical(zero));
        Vector<float> input(std::vector<float>{1, 2, 3}), output(std::vector<float>{7, 8});
        zeroPlan.multiply(input, output);
        EXPECT_EQ(linalg::detail::hostCopy(output), (std::vector<float>{0, 0}));
    }
    TEST_F(OptimizationGpu, DeviceFieldsAndRegionMatchIndependentInterfacesAndStreaming) {
        std::vector<thermal::P1CellSample> cells(517);
        for (std::size_t i = 0; i < cells.size(); ++i) {
            auto& c = cells[i];
            c.dimension_ = 2;
            c.volume_ = 0.5;
            c.capacity_ = 3;
            c.conductivity_ = 2;
            c.dofs_[0] = 0;
            c.dofs_[1] = 1;
            c.dofs_[2] = 2;
            c.gradients_[0] = -1;
            c.gradients_[1] = -1;
            c.gradients_[3] = 1;
            c.gradients_[7] = 1;
        }
        Vector<double> t(std::vector<double>{300, 302, 304});
        auto expected = thermal::evaluateP1Gradient(cells, t);
        std::vector<double> volumes(cells.size(), .5);
        auto reference = thermal::reduceRegion(expected, volumes);
        for (auto budget : {std::size_t{0}, std::size_t{1024 * 1024}}) {
            thermal::detail::P1PostprocessWorkspace work(cells, cells.size(), budget);
            auto actual = work.fields(0, cells.size(), t);
            ASSERT_EQ(actual.size(), expected.size());
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_DOUBLE_EQ(actual[i].gradient_[0], expected[i].gradient_[0]);
                EXPECT_DOUBLE_EQ(actual[i].gradient_[1], expected[i].gradient_[1]);
                EXPECT_DOUBLE_EQ(actual[i].flux_[0], -2 * expected[i].gradient_[0]);
                EXPECT_DOUBLE_EQ(actual[i].flux_[1], -2 * expected[i].gradient_[1]);
            }
            auto region = work.region(0, cells.size(), t);
            EXPECT_DOUBLE_EQ(region.minimum_, reference.minimum_);
            EXPECT_DOUBLE_EQ(region.maximum_, reference.maximum_);
            EXPECT_DOUBLE_EQ(region.integral_, reference.integral_);
            EXPECT_DOUBLE_EQ(region.energy_, reference.energy_);
            EXPECT_DOUBLE_EQ(region.volume_, reference.volume_);
            t.copyFromHost(std::vector<double>{310, 312, 314});
            EXPECT_DOUBLE_EQ(work.region(0, cells.size(), t).minimum_, 310);
            t.copyFromHost(std::vector<double>{300, 302, 304});
            EXPECT_THROW(work.region(0, 0, t), std::invalid_argument);
        }
        t.copyFromHost(std::vector<double>{300, std::numeric_limits<double>::infinity(), 304});
        thermal::detail::P1PostprocessWorkspace work(cells, cells.size(), 0);
        EXPECT_THROW(work.region(0, cells.size(), t), std::overflow_error);
    }
    TEST_F(OptimizationGpu, StepPlanPreservesOriginalDivisionOrderAndOldSystems) {
        auto spatial = assemblyfixture::matrix(3, {4, -1, -2, -1, 5, -1, -2, -1, 6});
        auto mass = assemblyfixture::matrix(3, {2, 1, .5, 1, 3, 1, .5, 1, 4});
        Vector<double> load(std::vector<double>{2, -3, 4}), previous(std::vector<double>{300, 310, 320});
        const std::vector<std::uint64_t> ids{0, 2};
        for (double dt : {.1, .3, 2.}) {
            systems::detail::BackwardEulerPlan plan(spatial, mass, dt, ids);
            std::optional<systems::LinearSystem> old;
            for (const auto& g : {std::vector<double>{300, 320}, std::vector<double>{310, 340}}) {
                auto expected = systems::applyDirichlet(
                    systems::buildBackwardEulerSystem(spatial, mass, load, previous, dt), ids, g);
                auto actual = plan.build(load, previous, g);
                EXPECT_EQ(linalg::detail::hostCopy(actual.matrix().values()),
                          linalg::detail::hostCopy(expected.matrix().values()));
                EXPECT_EQ(linalg::detail::hostCopy(actual.rhs()), linalg::detail::hostCopy(expected.rhs()));
                if (old) {
                    EXPECT_EQ(linalg::detail::hostCopy(old->rhs())[0], 300);
                }
                else {
                    old = actual;
                }
            }
            EXPECT_THROW(plan.build(load, previous, std::vector<double>{300}), std::invalid_argument);
            EXPECT_THROW(plan.build(load, previous, std::vector<double>{300, std::numeric_limits<double>::infinity()}),
                         std::invalid_argument);
        }
    }
    TEST_F(OptimizationGpu, AssemblyMappingCacheReusesExactSlicesAndNeverCommitsFailedCandidates) {
        auto source = assembly::buildAssemblyPlan(3, std::vector<assembly::DofList>{{0, 1}, {1, 2}, {0, 1}});
        for (auto budget : {std::size_t{0}, std::size_t{128}, std::size_t{1024 * 1024}}) {
            auto cache = std::make_shared<assembly::detail::DeviceAssemblyPlan>(source, budget);
            assembly::AssemblyAccumulator a(source, cache), b(source, cache);
            CudaBuffer<double> local(std::vector<double>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
            a.addMatrix(0, 3, local);
            b.addMatrix(0, 3, local);
            EXPECT_EQ(linalg::detail::hostCopy(a.matrix().values()), linalg::detail::hostCopy(b.matrix().values()));
            if (budget >= 1024 * 1024) {
                EXPECT_EQ(cache->mappingBuilds(), 1);
                EXPECT_EQ(cache->cacheHits(), 1);
            }
            else {
                EXPECT_EQ(cache->mappingBuilds(), 2);
            }
            const auto before = linalg::detail::hostCopy(a.matrix().values());
            local.copyFromHost(
                std::vector<double>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, std::numeric_limits<double>::infinity()});
            EXPECT_THROW(a.addMatrix(0, 3, local), std::overflow_error);
            EXPECT_EQ(linalg::detail::hostCopy(a.matrix().values()), before);
            CudaBuffer<double> part(std::vector<double>{1, 2, 3, 4});
            EXPECT_NO_THROW(a.addMatrix(1, 1, part));
            EXPECT_NO_THROW(b.addMatrix(1, 1, part));
            CudaBuffer<double> vector(std::vector<double>{1, 2});
            EXPECT_NO_THROW(a.addVector(1, 1, vector));
            EXPECT_THROW(a.addVector(3, 1, vector), std::out_of_range);
        }
    }

} // namespace
