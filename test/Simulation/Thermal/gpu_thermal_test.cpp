/** @file gpu_thermal_test.cpp
 * @brief 验证GPU批积分与独立主机公式一致，覆盖跨block、来源、预算及数值失败。
 */
#include <Simulation/Thermal/GpuElementOperations.cuh>
#include <cmath>
#include <gtest/gtest.h>
#include "support/IntegrationFixture.cuh"
namespace {
    using namespace emberfield;
    using namespace thermal;
    /** @brief GPU测试前确认设备可用；无设备时明确跳过。 */
    class GpuThermal : public testing::Test {
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
    TEST_F(GpuThermal, TriangleAndTetBatchesCrossBlocksMatchIndependentHostOperators) {
        for (unsigned d : {2U, 3U}) {
            auto m = integrationfixture::data(d);
            // 非对称Jacobian和负局部方向，避免只测单位J。
            m.nodes_[1].position_ = {2, 1, 0};
            m.nodes_[2].position_ = {3, 4, 0};
            std::swap(m.elements_[0].nodes_[0], m.elements_[0].nodes_[1]);
            fem::GeometryOptions o;
            if (d == 3) {
                o.frame_ = fem::GeometryFrame::SolidXYZ;
            }
            auto p = integrationfixture::geometry(m, o);
            std::vector<fem::CellIntegrationData> tables;
            for (std::size_t e = 0; e < 257; ++e) {
                tables.push_back(fem::evaluateCellIntegration(p, 0));
            }
            const auto batch = uploadCellBatch(tables);
            const auto q = batch.pointCount();
            std::vector<double> k(batch.size() * q), c(k.size()), source(k.size()),
                scales(batch.size(), d == 2 ? 0.2 : 1);
            for (std::size_t i = 0; i < k.size(); ++i) {
                k[i] = 1 + i % 7;
                c[i] = 20 + i % 5;
                source[i] = static_cast<int>(i % 11) - 5;
            }
            auto km = integrateConductivityBatch(batch, k, scales).copyToHost();
            auto mm = integrateCapacityBatch(batch, c, scales).copyToHost();
            auto fm = integrateVolumeSourceBatch(batch, source, scales).copyToHost();
            for (std::size_t e = 0; e < tables.size(); ++e) {
                compare(km, e, integrateConductivity(tables[e], std::span(k).subspan(e * q, q), scales[e]));
                compare(mm, e, integrateCapacity(tables[e], std::span(c).subspan(e * q, q), scales[e]));
                compare(fm, e, integrateVolumeSource(tables[e], std::span(source).subspan(e * q, q), scales[e]));
            }
        }
    }
    TEST_F(GpuThermal, EveryFacetBatchMatchesFluxAndConvectionIncludingZeroH) {
        for (unsigned d : {2U, 3U}) {
            fem::GeometryOptions o;
            if (d == 3) {
                o.frame_ = fem::GeometryFrame::SolidXYZ;
            }
            auto p = integrationfixture::geometry(integrationfixture::data(d), o);
            std::vector<fem::FacetIntegrationData> tables;
            for (std::size_t i = 0; i < p->topology().facets_.size(); ++i) {
                tables.push_back(fem::evaluateFacetIntegration(p, i));
            }
            auto batch = uploadFacetBatch(tables);
            auto q = batch.pointCount();
            std::vector<double> h(batch.size() * q), ambient(h.size()), flux(h.size()),
                scales(batch.size(), d == 2 ? 0.5 : 1);
            for (std::size_t i = 0; i < h.size(); ++i) {
                h[i] = i % 4;
                ambient[i] = 290 + i;
                flux[i] = i % 2 ? 7 : -3;
            }
            auto cg = integrateConvectionBatch(batch, h, ambient, scales);
            auto matrix = cg.matrix().copyToHost(), load = cg.load().copyToHost();
            auto fg = integrateOutwardFluxBatch(batch, flux, scales).copyToHost();
            for (std::size_t e = 0; e < tables.size(); ++e) {
                auto ref = integrateConvection(tables[e], std::span(h).subspan(e * q, q),
                                               std::span(ambient).subspan(e * q, q), scales[e]);
                compare(matrix, e, ref.matrix());
                compare(load, e, ref.load());
                compare(fg, e, integrateOutwardFlux(tables[e], std::span(flux).subspan(e * q, q), scales[e]));
            }
        }
    }
    TEST_F(GpuThermal, ShapesProvenancePrecisionAndBudgetsAreEnforced) {
        auto p = integrationfixture::geometry();
        auto other = integrationfixture::geometry();
        std::vector<fem::CellIntegrationData> tables{fem::evaluateCellIntegration(p, 0)};
        auto batch = uploadCellBatch(tables);
        std::vector<double> k(3, 1), s(1, 1);
        EXPECT_THROW((void)integrateConductivityBatch(batch, {}, s), std::invalid_argument);
        k[0] = -1;
        EXPECT_THROW((void)integrateConductivityBatch(batch, k, s), std::invalid_argument);
        k[0] = 1;
        EXPECT_THROW((void)integrateOutwardFluxBatch(batch, k, s), std::invalid_argument);
        auto low = std::vector<fem::CellIntegrationData>{fem::evaluateCellIntegration(p, 0, 1)};
        auto lowBatch = uploadCellBatch(low);
        EXPECT_THROW((void)integrateCapacityBatch(lowBatch, std::vector<double>{1}, s), std::invalid_argument);
        tables.push_back(fem::evaluateCellIntegration(other, 0));
        EXPECT_THROW((void)uploadCellBatch(tables), std::invalid_argument);
        EXPECT_THROW((void)uploadCellBatch(tables, {1, 1000}), std::length_error);
        EXPECT_THROW((void)uploadCellBatch({}, {}), std::invalid_argument);
        tables.pop_back();
        EXPECT_THROW((void)uploadCellBatch(tables, {1, 8}), std::length_error);
        auto a = integrateVolumeSourceBatch(batch, k, s);
        auto b = uploadCellBatch(tables);
        auto c = integrateVolumeSourceBatch(b, k, s);
        EXPECT_THROW((void)addLocalVectorBatches(batch, a, c), std::invalid_argument);
        auto sum = addLocalVectorBatches(batch, a, a).copyToHost();
        for (double x : sum) {
            EXPECT_NEAR(x, 1.0 / 3, 1e-14);
        }
    }
    TEST_F(GpuThermal, NumericalErrorsDoNotPublishPartialBatchesAndPreserveInputs) {
        auto p = integrationfixture::geometry();
        std::vector<fem::CellIntegrationData> t{fem::evaluateCellIntegration(p, 0)};
        auto b = uploadCellBatch(t);
        std::vector<double> s{1e308}, k(3, 1e308);
        EXPECT_THROW((void)integrateConductivityBatch(b, k, s), fem::EvaluationError);
        EXPECT_DOUBLE_EQ(k[0], 1e308);
        EXPECT_THROW((void)integrateCapacityBatch(b, std::vector<double>(3, 1e-300), std::vector<double>{1e-300}),
                     fem::EvaluationError);
        auto good = integrateConductivityBatch(b, std::vector<double>(3, 1), std::vector<double>{1}).copyToHost();
        EXPECT_NEAR(good[0], 1, 1e-14);
    }
    TEST_F(GpuThermal, ScaledProductAvoidsUnnecessaryIntermediateOverflow) {
        auto p = integrationfixture::geometry();
        std::vector<fem::CellIntegrationData> t{fem::evaluateCellIntegration(p, 0)};
        auto b = uploadCellBatch(t);
        auto result =
            integrateVolumeSourceBatch(b, std::vector<double>(3, 1e-308), std::vector<double>{1e308}).copyToHost();
        for (auto x : result) {
            EXPECT_NEAR(x, 1.0 / 6, 1e-14);
        }
    }
} // namespace
