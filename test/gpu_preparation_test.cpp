/** @file gpu_preparation_test.cpp
 * @brief 验证算例配置到独立GPU局部热算子的有序适配。
 */
#include <Case/GpuPreparation.cuh>
#include <cmath>
#include <gtest/gtest.h>
#include "support/CaseFixture.cuh"
namespace {
    using namespace thermofem;
    using namespace thermal;
    /** @brief GPU测试前确认设备可用；无设备时明确跳过。 */
    class GpuPreparation : public testing::Test {
    protected:
        /** @brief 建立设备前提；无参数无返回值。@pre 无。@code SetUp(); @endcode */
        void SetUp() override {
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
                GTEST_SKIP() << "CUDA设备不可用";
            }
        }
    };
    /** @brief 对比一个实体矩阵与批条目。@param values 回读批。@param e 行。@param ref 主机矩阵。
     * @pre 维数一致。@note 无返回，逐项断言。@code compare(values,e,ref); @endcode */
    void compare(const std::vector<double>& values, std::size_t e, const LocalMatrix& ref) {
        const auto width = ref.size() * ref.size();
        for (std::size_t i = 0; i < width; ++i) {
            EXPECT_NEAR(values[e * width + i], ref.entries()[i], 1e-11 * std::max(1.0, std::abs(ref.entries()[i])));
        }
    }
    /** @brief 对比一个实体载荷与批条目。@param values 批。@param e 行。@param ref 主机载荷。
     * @pre 同序同维。@note 无返回。@code compare(values,e,ref); @endcode */
    void compare(const std::vector<double>& values, std::size_t e, const LocalVector& ref) {
        for (std::size_t i = 0; i < ref.size(); ++i) {
            EXPECT_NEAR(values[e * ref.size() + i], ref[i], 1e-11 * std::max(1.0, std::abs(ref[i])));
        }
    }
    TEST_F(GpuPreparation, PreparedCaseBatchesPreserveRequestedRowsAndSourceOrder) {
        auto c = casefixture::definition();
        casefixture::makeTransient(c);
        c.sources_ = {{"a", casefixture::group(2, 11), problem::ConstantScalar{10}},
                      {"b", casefixture::group(2, 11), problem::ConstantScalar{-3}}};
        auto binding = casefixture::bind(c);
        ASSERT_TRUE(binding.bound_);
        auto space = problem::buildTemperatureSpace(*binding.bound_);
        auto r = problem::prepareThermalCase(binding.bound_, space.space_);
        ASSERT_TRUE(r.ok());
        std::vector<std::size_t> rows{1, 0, 1};
        auto batch = problem::evaluateCellBatch(*r.prepared(), rows);
        EXPECT_EQ(batch.cellRows(), rows);
        ASSERT_TRUE(batch.capacity());
        auto k = batch.conductivity().copyToHost(), m = batch.capacity()->copyToHost(), f = batch.load().copyToHost();
        for (std::size_t e = 0; e < rows.size(); ++e) {
            auto ref = problem::evaluateCellTerms(*r.prepared(), rows[e]);
            compare(k, e, ref.conductivity());
            compare(m, e, *ref.capacity());
            compare(f, e, ref.load());
        }
        EXPECT_THROW((void)problem::evaluateCellBatch(*r.prepared(), rows, {1, 10000}), std::length_error);
    }
    TEST_F(GpuPreparation, PreparedBoundaryBatchDoesNotMergeHAndAmbientAndRejectsMixedTypes) {
        auto c = casefixture::definition();
        c.naturalOverlap_ = problem::NaturalOverlap::Add;
        c.boundaries_.push_back(
            {"a", casefixture::group(1, 21),
             problem::ConvectionBoundary{problem::ConstantScalar{2}, problem::ConstantScalar{300}}});
        c.boundaries_.push_back(
            {"b", casefixture::group(1, 21),
             problem::ConvectionBoundary{problem::ConstantScalar{3}, problem::ConstantScalar{400}}});
        c.boundaries_.push_back({"flux", casefixture::group(1, 22), problem::FluxBoundary{problem::ConstantScalar{5}}});
        auto b = casefixture::bind(c);
        auto s = problem::buildTemperatureSpace(*b.bound_);
        auto r = problem::prepareThermalCase(b.bound_, s.space_);
        ASSERT_TRUE(r.ok());
        std::vector<std::size_t> convectionRows, fluxRows;
        const auto& terms = r.prepared()->conditions()->naturalTerms();
        for (std::size_t i = 0; i < terms.size(); ++i) {
            if (std::holds_alternative<problem::FluxBoundary>(c.boundaries_[terms[i].definitionIndex()].value_)) {
                fluxRows.push_back(i);
            }
            else {
                convectionRows.push_back(i);
            }
        }
        auto batch = problem::evaluateBoundaryBatch(*r.prepared(), convectionRows);
        const auto& cg = std::get<GpuConvectionBatch>(batch.contribution());
        auto matrices = cg.matrix().copyToHost(), loads = cg.load().copyToHost();
        for (std::size_t e = 0; e < convectionRows.size(); ++e) {
            auto ref =
                std::get<ConvectionContribution>(problem::evaluateBoundaryTerm(*r.prepared(), convectionRows[e]));
            compare(matrices, e, ref.matrix());
            compare(loads, e, ref.load());
        }
        auto flux = problem::evaluateBoundaryBatch(*r.prepared(), fluxRows);
        compare(std::get<GpuLocalVectorBatch>(flux.contribution()).copyToHost(), 0,
                std::get<LocalVector>(problem::evaluateBoundaryTerm(*r.prepared(), fluxRows[0])));
        convectionRows.push_back(fluxRows[0]);
        EXPECT_THROW((void)problem::evaluateBoundaryBatch(*r.prepared(), convectionRows), std::invalid_argument);
    }
} // namespace
