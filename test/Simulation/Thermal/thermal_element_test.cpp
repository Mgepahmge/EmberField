/** @file thermal_element_test.cpp
 * @brief 以解析矩阵、单位尺度和错误输入验证独立热运算。
 */
#include <Simulation/Thermal/ElementOperations.cuh>
#include <cmath>
#include <gtest/gtest.h>
#include <numeric>
#include "support/IntegrationFixture.cuh"
namespace {
    using namespace emberfield;
    using namespace thermal;
    using namespace integrationfixture;
    /** @brief 常量点样本。@param q 点数。@param v 值。@pre 无。
     * @return 拥有数组。@code auto a=sample(3,2); @endcode */
    std::vector<double> sample(std::size_t q, double v) { return std::vector<double>(q, v); }
    TEST(ThermalElement, TriangleAnalyticMatricesAndSource) {
        const auto t = fem::evaluateCellIntegration(geometry(data()), 0);
        auto k = integrateConductivity(t, sample(t.pointCount(), 6), 2);
        auto m = integrateCapacity(t, sample(t.pointCount(), 12), 2);
        auto f = integrateVolumeSource(t, sample(t.pointCount(), 9), 2);
        const double gram[9] = {2, -1, -1, -1, 1, 0, -1, 0, 1};
        for (std::size_t a = 0; a < 3; ++a) {
            EXPECT_NEAR(f[a], 3, 1e-14);
            double sum = 0;
            for (std::size_t b = 0; b < 3; ++b) {
                EXPECT_NEAR(k(a, b), 6 * gram[a * 3 + b], 1e-13);
                EXPECT_NEAR(m(a, b), a == b ? 2 : 1, 1e-14);
                EXPECT_DOUBLE_EQ(k(a, b), k(b, a));
                EXPECT_DOUBLE_EQ(m(a, b), m(b, a));
                sum += k(a, b);
            }
            EXPECT_NEAR(sum, 0, 1e-13);
        }
    }
    TEST(ThermalElement, TetrahedronAnalyticMatricesAndSource) {
        fem::GeometryOptions o;
        o.frame_ = fem::GeometryFrame::SolidXYZ;
        auto t = fem::evaluateCellIntegration(geometry(data(3), o), 0);
        auto k = integrateConductivity(t, sample(t.pointCount(), 6), 1);
        auto m = integrateCapacity(t, sample(t.pointCount(), 120), 1);
        auto f = integrateVolumeSource(t, sample(t.pointCount(), 24), 1);
        for (std::size_t a = 0; a < 4; ++a) {
            EXPECT_NEAR(f[a], 1, 1e-14);
            for (std::size_t b = 0; b < 4; ++b) {
                const double gram = a == 0 && b == 0 ? 3 : (a == 0 || b == 0) ? -1 : a == b ? 1 : 0;
                EXPECT_NEAR(k(a, b), gram, 1e-13);
                EXPECT_NEAR(m(a, b), a == b ? 2 : 1, 1e-13);
            }
        }
    }
    TEST(ThermalElement, BoundaryFluxSignAndConvectionEquilibrium) {
        auto p = geometry(data());
        for (std::size_t side = 0; side < p->topology().facets_.size(); ++side) {
            auto t = fem::evaluateFacetIntegration(p, side);
            const double length = p->exteriorFacet(side).measure();
            auto f = integrateOutwardFlux(t, sample(t.pointCount(), 8), 2);
            auto c = integrateConvection(t, sample(t.pointCount(), 3), sample(t.pointCount(), 7), 2);
            for (std::size_t a = 0; a < 2; ++a) {
                EXPECT_NEAR(f[a], -8 * length, 1e-13);
                EXPECT_NEAR(c.load()[a], 21 * length, 1e-13);
                double equilibrium = 0;
                for (std::size_t b = 0; b < 2; ++b) {
                    EXPECT_NEAR(c.matrix()(a, b), length * (a == b ? 2 : 1), 1e-13);
                    equilibrium += 7 * c.matrix()(a, b);
                }
                EXPECT_NEAR(equilibrium, c.load()[a], 1e-12);
            }
        }
    }
    TEST(ThermalElement, TetrahedronFaceUsesAreaOnce) {
        fem::GeometryOptions o;
        o.frame_ = fem::GeometryFrame::SolidXYZ;
        auto p = geometry(data(3), o);
        for (std::size_t side = 0; side < p->topology().facets_.size(); ++side) {
            auto t = fem::evaluateFacetIntegration(p, side);
            auto c = integrateConvection(t, sample(t.pointCount(), 12), sample(t.pointCount(), 2), 1);
            const double area = p->exteriorFacet(side).measure();
            for (std::size_t a = 0; a < 3; ++a) {
                EXPECT_NEAR(c.load()[a], 8 * area, 1e-13);
                for (std::size_t b = 0; b < 3; ++b) {
                    EXPECT_NEAR(c.matrix()(a, b), area * (a == b ? 2 : 1), 1e-13);
                }
            }
        }
    }
    TEST(ThermalElement, UnderintegratedMassAndConvectionAreRejected) {
        auto p = geometry(data());
        auto t = fem::evaluateCellIntegration(p, 0, 1);
        auto f = fem::evaluateFacetIntegration(p, 0, 1);
        EXPECT_THROW((void)integrateCapacity(t, sample(t.pointCount(), 1), 1), std::invalid_argument);
        EXPECT_THROW((void)integrateConvection(f, sample(f.pointCount(), 1), sample(f.pointCount(), 1), 1),
                     std::invalid_argument);
        EXPECT_NO_THROW((void)integrateConductivity(t, sample(t.pointCount(), 1), 1));
    }
    TEST(ThermalElement, InvalidSamplesAndSectionAreRejected) {
        auto p = geometry(data());
        auto t = fem::evaluateCellIntegration(p, 0);
        auto side = fem::evaluateFacetIntegration(p, 0);
        EXPECT_THROW((void)integrateConductivity(t, {}, 1), std::invalid_argument);
        for (double v :
             {-1.0, 0.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
            EXPECT_THROW((void)integrateConductivity(t, sample(t.pointCount(), v), 1), std::invalid_argument);
            EXPECT_THROW((void)integrateCapacity(t, sample(t.pointCount(), v), 1), std::invalid_argument);
        }
        EXPECT_THROW((void)integrateConvection(side, sample(side.pointCount(), 0), sample(side.pointCount(), -1), 1),
                     std::invalid_argument);
        EXPECT_THROW((void)integrateVolumeSource(t, sample(t.pointCount(), 1), 0), std::invalid_argument);
        EXPECT_THROW((void)integrateConvection(side, sample(side.pointCount(), -1), sample(side.pointCount(), 300), 1),
                     std::invalid_argument);
        fem::GeometryOptions o;
        o.frame_ = fem::GeometryFrame::SolidXYZ;
        auto tet = fem::evaluateCellIntegration(geometry(data(3), o), 0);
        EXPECT_THROW((void)integrateConductivity(tet, sample(tet.pointCount(), 1), 2), std::invalid_argument);
    }
    TEST(ThermalElement, ZeroAndSignedSourcesAreLegalAndOutputIndicesChecked) {
        auto p = geometry(data());
        auto t = fem::evaluateCellIntegration(p, 0);
        auto side = fem::evaluateFacetIntegration(p, 0);
        auto f = integrateVolumeSource(t, sample(t.pointCount(), 0), 1);
        auto neg = integrateVolumeSource(t, sample(t.pointCount(), -6), 1);
        auto zero = integrateConvection(side, sample(side.pointCount(), 0), sample(side.pointCount(), 300), 1);
        EXPECT_EQ(f.size(), 3);
        EXPECT_EQ(f.entries(), sample(3, 0));
        EXPECT_NEAR(std::accumulate(neg.entries().begin(), neg.entries().end(), 0.0), -3, 1e-14);
        EXPECT_EQ(zero.matrix().entries(), sample(4, 0));
        EXPECT_THROW((void)f[99], std::out_of_range);
        EXPECT_THROW((void)zero.matrix()(0, 2), std::out_of_range);
    }
    TEST(ThermalElement, ThicknessAndSpatialScaleHaveCorrectPowers) {
        auto m = data();
        auto original = fem::evaluateCellIntegration(geometry(m), 0);
        for (auto& n : m.nodes_) {
            n.position_[0] *= 3;
            n.position_[1] *= 3;
        }
        auto scaled = fem::evaluateCellIntegration(geometry(m), 0);
        auto k = integrateConductivity(original, sample(3, 1), 2);
        auto ks = integrateConductivity(scaled, sample(3, 1), 2);
        auto mass = integrateCapacity(original, sample(3, 1), 2);
        auto ms = integrateCapacity(scaled, sample(3, 1), 2);
        auto kd = integrateConductivity(original, sample(3, 1), 4);
        for (std::size_t i = 0; i < 9; ++i) {
            EXPECT_NEAR(ks.entries()[i], k.entries()[i], 1e-13);
            EXPECT_NEAR(ms.entries()[i], 9 * mass.entries()[i], 1e-13);
            EXPECT_NEAR(kd.entries()[i], 2 * k.entries()[i], 1e-13);
        }
    }
    TEST(ThermalElement, VariableSamplesFollowQuadratureOrder) {
        auto t = fem::evaluateCellIntegration(geometry(data()), 0);
        std::vector<double> source{1, 2, 4};
        auto f = integrateVolumeSource(t, source, 1);
        for (std::size_t a = 0; a < 3; ++a) {
            double expected = 0;
            for (std::size_t q = 0; q < 3; ++q) {
                expected += t.physicalWeights()[q] * source[q] * t.values()[q * 3 + a];
            }
            EXPECT_NEAR(f[a], expected, 1e-14);
        }
    }
    TEST(ThermalElement, OverflowAndPositiveMatrixUnderflowAreReported) {
        auto t = fem::evaluateCellIntegration(geometry(data()), 0);
        EXPECT_THROW((void)integrateConductivity(t, sample(3, 1e308), 1e308), fem::EvaluationError);
        EXPECT_THROW((void)integrateCapacity(t, sample(3, 1e-300), 1e-300), fem::EvaluationError);
        EXPECT_THROW((void)integrateVolumeSource(t, sample(3, 1e308), 1e308), fem::EvaluationError);
    }
} // namespace
