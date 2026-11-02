/** @file reference_element_test.cpp
 * @brief 以插值恒等式、仿射再现和侧限制验证参考有限元。
 */
#include <FEM/Element/ReferenceElement.cuh>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <stdexcept>

namespace {
    using namespace thermofem;
    TEST(ReferenceElement, NodalInterpolationAndEmptyBatch) {
        for (auto family : {mesh::CellFamily::Triangle, mesh::CellFamily::Tetrahedron}) {
            const fem::ElementKey key{family};
            const auto& e = fem::referenceElement(key);
            const auto table = fem::evaluateBasis(key, e.interpolationPoints());
            const auto n = e.dofCount();
            EXPECT_EQ(table.values_.size(), n * n);
            for (std::size_t q = 0; q < n; ++q) {
                for (std::size_t a = 0; a < n; ++a) {
                    EXPECT_DOUBLE_EQ(table.values_[q * n + a], q == a ? 1 : 0);
                }
            }
            const auto empty = fem::evaluateBasis(key, {});
            EXPECT_EQ(empty.dimension_, e.dimension());
            EXPECT_EQ(empty.dofCount_, n);
            EXPECT_TRUE(empty.values_.empty());
            EXPECT_TRUE(empty.gradients_.empty());
        }
    }
    TEST(ReferenceElement, ReproducesAffineValuesGradientsAndPartitionOfUnity) {
        std::mt19937 engine(1729);
        std::uniform_real_distribution<double> random(-1, 2);
        for (auto family : {mesh::CellFamily::Triangle, mesh::CellFamily::Tetrahedron}) {
            const fem::ElementKey key{family};
            const auto& e = fem::referenceElement(key);
            const auto n = e.dofCount();
            const auto d = e.dimension();
            const std::array<double, 3> slope{2, -3, 4};
            for (int trial = 0; trial < 40; ++trial) {
                std::array<double, 3> p{random(engine), random(engine), d == 3 ? random(engine) : 0};
                const auto t = fem::evaluateBasis(key, std::span(&p, 1));
                double total = 0;
                double value = 0;
                std::array<double, 3> gradient{}, gradientSum{};
                double exact = 7;
                for (unsigned k = 0; k < d; ++k) {
                    exact += slope[k] * p[k];
                }
                for (std::size_t a = 0; a < n; ++a) {
                    double nodal = 7;
                    for (unsigned k = 0; k < d; ++k) {
                        nodal += slope[k] * e.interpolationPoints()[a][k];
                    }
                    total += t.values_[a];
                    value += nodal * t.values_[a];
                    for (unsigned k = 0; k < d; ++k) {
                        gradient[k] += nodal * t.gradients_[a * d + k];
                        gradientSum[k] += t.gradients_[a * d + k];
                    }
                }
                EXPECT_NEAR(total, 1, 2e-15);
                EXPECT_NEAR(value, exact, 2e-14);
                for (unsigned k = 0; k < d; ++k) {
                    EXPECT_DOUBLE_EQ(gradient[k], slope[k]);
                    EXPECT_DOUBLE_EQ(gradientSum[k], 0);
                    auto left = p;
                    auto right = p;
                    left[k] -= 1e-5;
                    right[k] += 1e-5;
                    const auto l = fem::evaluateBasis(key, std::span(&left, 1));
                    const auto r = fem::evaluateBasis(key, std::span(&right, 1));
                    for (std::size_t a = 0; a < n; ++a) {
                        EXPECT_NEAR((r.values_[a] - l.values_[a]) / 2e-5, t.gradients_[a * d + k], 1e-10);
                    }
                }
            }
        }
    }
    TEST(ReferenceElement, FacetTraceMatchesMotherBasis) {
        for (auto family : {mesh::CellFamily::Triangle, mesh::CellFamily::Tetrahedron}) {
            const fem::ElementKey key{family};
            const auto& e = fem::referenceElement(key);
            for (std::size_t f = 0; f < e.facetCount(); ++f) {
                const auto dofs = e.facetLocalDofs(f);
                const std::vector<double> weights =
                    e.dimension() == 2 ? std::vector<double>{0.3, 0.7} : std::vector<double>{0.5, 0.2, 0.3};
                std::array<double, 3> p{};
                std::vector<double> exact(e.dofCount(), 0);
                for (std::size_t a = 0; a < dofs.size(); ++a) {
                    exact[dofs[a]] = weights[a];
                    for (unsigned k = 0; k < e.dimension(); ++k) {
                        p[k] += weights[a] * e.interpolationPoints()[dofs[a]][k];
                    }
                }
                const auto table = fem::evaluateBasis(key, std::span(&p, 1));
                for (std::size_t a = 0; a < e.dofCount(); ++a) {
                    EXPECT_NEAR(table.values_[a], exact[a], 1e-15);
                }
            }
            EXPECT_THROW((void)e.facetLocalDofs(e.facetCount()), std::out_of_range);
        }
    }
    TEST(ReferenceElement, CenterAndOrderedSideLayout) {
        const fem::ElementKey triangle{mesh::CellFamily::Triangle};
        const fem::ElementKey tetrahedron{mesh::CellFamily::Tetrahedron};
        std::array<double, 3> center{1. / 3, 1. / 3, 0};
        for (double value : fem::evaluateBasis(triangle, std::span(&center, 1)).values_) {
            EXPECT_NEAR(value, 1. / 3, 1e-15);
        }
        center = {0.25, 0.25, 0.25};
        for (double value : fem::evaluateBasis(tetrahedron, std::span(&center, 1)).values_) {
            EXPECT_DOUBLE_EQ(value, 0.25);
        }
        auto side = fem::referenceElement(triangle).facetLocalDofs(2);
        EXPECT_EQ((std::vector<std::size_t>(side.begin(), side.end())), (std::vector<std::size_t>{2, 0}));
        side = fem::referenceElement(tetrahedron).facetLocalDofs(0);
        EXPECT_EQ((std::vector<std::size_t>(side.begin(), side.end())), (std::vector<std::size_t>{0, 2, 1}));
    }
    TEST(ReferenceElement, RejectsInvalidKeysCoordinatesAndOverflow) {
        EXPECT_THROW((void)fem::referenceElement({mesh::CellFamily::Quadrilateral}), std::invalid_argument);
        EXPECT_THROW((void)fem::referenceElement({mesh::CellFamily::Triangle, fem::BasisFamily::ContinuousLagrange, 2}),
                     std::invalid_argument);
        EXPECT_THROW((void)fem::referenceElement({mesh::CellFamily::Triangle, static_cast<fem::BasisFamily>(99), 1}),
                     std::invalid_argument);
        const fem::ElementKey key{mesh::CellFamily::Triangle};
        for (double bad : {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
            for (unsigned k = 0; k < 3; ++k) {
                std::array<double, 3> p{};
                p[k] = bad;
                EXPECT_THROW(fem::evaluateBasis(key, std::span(&p, 1)), std::invalid_argument);
            }
        }
        std::array<double, 3> p{0, 0, 1};
        EXPECT_THROW(fem::evaluateBasis(key, std::span(&p, 1)), std::invalid_argument);
        p = {-std::numeric_limits<double>::max(), -std::numeric_limits<double>::max(), 0};
        EXPECT_THROW(fem::evaluateBasis(key, std::span(&p, 1)), std::overflow_error);
    }
} // namespace
