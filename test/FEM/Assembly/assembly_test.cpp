/** @file assembly_test.cpp
 * @brief 独立符号图、分量及GPU稳定散加的正确性与失败保护测试。
 */
#include <limits>
#include <numeric>
#include "support/AssemblyFixture.cuh"
namespace {
    using namespace emberfield;
    using namespace assembly;
    using namespace assemblyfixture;
    TEST(AssemblyPlan, CanonicalGraphSlotsAndVertexComponents) {
        std::vector<DofList> s{{2, 0}, {1, 2}, {2, 0}, {4}};
        auto p = buildAssemblyPlan(6, s);
        EXPECT_EQ(p->rowOffsets(), (std::vector<Index>{0, 2, 4, 7, 8, 9, 10}));
        EXPECT_EQ(p->columns(), (std::vector<Index>{0, 2, 1, 2, 0, 1, 2, 3, 4, 5}));
        EXPECT_EQ(p->components(), (std::vector<std::size_t>{0, 0, 0, 1, 2, 3}));
        EXPECT_EQ(p->componentCount(), 4);
        EXPECT_EQ(p->matrixSlots()[0], 6);
        EXPECT_EQ(p->matrixSlots()[1], 4);
        EXPECT_EQ(p->matrixOffsets().back(), 13);
        EXPECT_EQ(p->vectorOffsets().back(), 7);
        auto isolated = buildAssemblyPlan(3, {});
        EXPECT_EQ(isolated->columns(), (std::vector<Index>{0, 1, 2}));
        EXPECT_EQ(isolated->componentCount(), 3);
    }
    TEST(AssemblyPlan, RejectsInvalidTopologyAndEveryBudget) {
        EXPECT_THROW((void)buildAssemblyPlan(0, {}), std::invalid_argument);
        std::vector<DofList> empty{{}}, duplicate{{0, 0}}, outside{{2}}, valid{{0, 1}};
        EXPECT_THROW((void)buildAssemblyPlan(2, empty), std::invalid_argument);
        EXPECT_THROW((void)buildAssemblyPlan(2, duplicate), std::invalid_argument);
        EXPECT_THROW((void)buildAssemblyPlan(2, outside), std::out_of_range);
        auto l = AssemblyLimits{};
        l.maxDofs_ = 1;
        EXPECT_THROW((void)buildAssemblyPlan(2, valid, l), std::length_error);
        l = AssemblyLimits{};
        l.maxLocalEntries_ = 3;
        EXPECT_THROW((void)buildAssemblyPlan(2, valid, l), std::length_error);
        l = AssemblyLimits{};
        l.maxNonzeros_ = 3;
        EXPECT_THROW((void)buildAssemblyPlan(2, valid, l), std::length_error);
        l = AssemblyLimits{};
        l.maxStencils_ = 1;
        valid.push_back({1});
        EXPECT_THROW((void)buildAssemblyPlan(2, valid, l), std::length_error);
        l.maxStencils_ = 0;
        EXPECT_THROW((void)buildAssemblyPlan(2, {}, l), std::invalid_argument);
    }
    TEST_F(GpuAssembly, RepeatedStencilsAndBatchesMatchDenseAndPreserveSymmetry) {
        std::vector<DofList> s;
        for (std::size_t i = 0; i < 300; ++i) {
            s.push_back({i, i + 1});
        }
        s.push_back({0, 1});
        auto p = buildAssemblyPlan(302, s);
        std::vector<double> k(s.size() * 4), f(s.size() * 2), expected(302 * 302, 0), b(302, 0);
        for (std::size_t e = 0; e < s.size(); ++e) {
            const double t = 1 + e % 9;
            k[4 * e] = k[4 * e + 3] = 2 * t;
            k[4 * e + 1] = k[4 * e + 2] = -t;
            f[2 * e] = t;
            f[2 * e + 1] = -2 * t;
            for (std::size_t i = 0; i < 2; ++i) {
                b[s[e][i]] += f[2 * e + i];
                for (std::size_t j = 0; j < 2; ++j) {
                    expected[s[e][i] * 302 + s[e][j]] += k[4 * e + 2 * i + j];
                }
            }
        }
        AssemblyAccumulator a(p), second(p), split(p);
        CudaBuffer<double> kd{std::span<const double>(k)}, fd{std::span<const double>(f)};
        a.addMatrix(0, s.size(), kd);
        a.addVector(0, s.size(), fd);
        second.addMatrix(0, s.size(), kd);
        second.addVector(0, s.size(), fd);
        for (std::size_t e = 0; e < s.size(); ++e) {
            CudaBuffer<double> ke{std::span<const double>(k).subspan(4 * e, 4)},
                fe{std::span<const double>(f).subspan(2 * e, 2)};
            split.addMatrix(e, 1, ke);
            split.addVector(e, 1, fe);
        }
        EXPECT_EQ(dense(a.matrix()), expected);
        EXPECT_EQ(host(a.load().values()), b);
        EXPECT_EQ(host(a.matrix().values()), host(second.matrix().values()));
        EXPECT_EQ(host(a.matrix().values()), host(split.matrix().values()));
        EXPECT_EQ(host(kd), k);
        EXPECT_EQ(host(fd), f);
        auto d = dense(a.matrix());
        for (std::size_t i = 0; i < 302; ++i) {
            for (std::size_t j = 0; j < 302; ++j) {
                EXPECT_EQ(d[i * 302 + j], d[j * 302 + i]);
            }
        }
        auto result = std::move(a).release();
        EXPECT_EQ(result.matrix_.rowCount(), 302);
        EXPECT_THROW(a.addMatrix(0, s.size(), kd), std::logic_error);
    }
    TEST_F(GpuAssembly, ValidatesSlicesPoliciesAndPreservesStateOnNumericalFailure) {
        auto p = buildAssemblyPlan(2, std::vector<DofList>{{0, 1}});
        AssemblyAccumulator a(p);
        std::vector<double> one{1, 2, 2, 3};
        CudaBuffer<double> v{std::span<const double>(one)};
        a.addMatrix(0, 1, v);
        const auto before = host(a.matrix().values());
        EXPECT_THROW(a.addMatrix(0, 0, v), std::out_of_range);
        EXPECT_THROW(a.addMatrix(1, 1, v), std::out_of_range);
        CudaBuffer<double> shortBuffer(1);
        EXPECT_THROW(a.addMatrix(0, 1, shortBuffer), std::invalid_argument);
        CudaBuffer<double> pinned(std::span<const double>(one), PinnedMemory{});
        EXPECT_THROW(a.addMatrix(0, 1, pinned), std::invalid_argument);
        one[2] = std::numeric_limits<double>::quiet_NaN();
        v.copyFromHost(one);
        EXPECT_THROW(a.addMatrix(0, 1, v), std::overflow_error);
        EXPECT_EQ(host(a.matrix().values()), before);
        one[2] = std::numeric_limits<double>::infinity();
        v.copyFromHost(one);
        EXPECT_THROW(a.addMatrix(0, 1, v), std::overflow_error);
        EXPECT_EQ(host(a.matrix().values()), before);
        std::fill(one.begin(), one.end(), std::numeric_limits<double>::max());
        v.copyFromHost(one);
        a.addMatrix(0, 1, v);
        auto maximum = host(a.matrix().values());
        EXPECT_THROW(a.addMatrix(0, 1, v), std::overflow_error);
        EXPECT_EQ(host(a.matrix().values()), maximum);
        std::vector<double> b{std::numeric_limits<double>::max(), 1};
        CudaBuffer<double> vb{std::span<const double>(b)};
        a.addVector(0, 1, vb);
        const auto original = host(a.load().values());
        EXPECT_THROW(a.addVector(0, 1, vb), std::overflow_error);
        EXPECT_EQ(host(a.load().values()), original);
    }
} // namespace
