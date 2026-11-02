/** @file geometry_integration_test.cpp
 * @brief 用解析变换、矩条件、局部侧序及失败边界验证几何和积分。
 */
#include <algorithm>
#include <cmath>
#include <future>
#include <gtest/gtest.h>
#include <numeric>
#include "support/IntegrationFixture.cuh"

namespace {
    using namespace emberfield;
    using namespace fem;
    using namespace integrationfixture;
    /** @brief 取得几何失败代码。@param m 数据。@param o 策略。@pre 夹具结构合法。
     * @return 失败代码；意外成功为测试失败。@code auto c=error(data()); @endcode */
    EvaluationCode error(mesh::MeshData m, GeometryOptions o = {}) {
        try {
            auto unused = geometry(std::move(m), o);
            ADD_FAILURE() << "意外几何成功";
        }
        catch (const EvaluationError& e) {
            return e.code();
        }
        return EvaluationCode::CoordinateRange;
    }
    TEST(Geometry, UnitTriangleAndTetrahedron) {
        for (unsigned d : {2U, 3U}) {
            GeometryOptions o;
            if (d == 3) {
                o.frame_ = GeometryFrame::SolidXYZ;
            }
            auto p = geometry(data(d), o);
            const auto& g = p->cells()[0];
            EXPECT_DOUBLE_EQ(g.measure(), d == 2 ? 0.5 : 1.0 / 6);
            EXPECT_DOUBLE_EQ(g.absoluteDeterminant(), 1);
            EXPECT_EQ(g.orientation(), 1);
            for (unsigned i = 0; i < d; ++i) {
                for (unsigned j = 0; j < d; ++j) {
                    EXPECT_DOUBLE_EQ(g.jacobian()[i * d + j], i == j ? 1 : 0);
                    EXPECT_DOUBLE_EQ(g.inverseJacobian()[i * d + j], i == j ? 1 : 0);
                }
            }
        }
    }
    TEST(Geometry, NonsymmetricShearUsesInverseTranspose) {
        auto m = data();
        m.nodes_[0].position_ = {5, -3, 0};
        m.nodes_[1].position_ = {7, -2, 0};
        m.nodes_[2].position_ = {8, 1, 0};
        auto p = geometry(m);
        auto t = evaluateCellIntegration(p, 0);
        // J=[[2,3],[1,4]]；逆转置列给物理基梯度。
        const std::vector<double> expected{-0.6, 0.2, 0.8, -0.6, -0.2, 0.4};
        for (std::size_t q = 0; q < t.pointCount(); ++q) {
            for (std::size_t i = 0; i < expected.size(); ++i) {
                EXPECT_NEAR(t.gradients()[q * 6 + i], expected[i], 1e-14);
            }
            EXPECT_NEAR(std::accumulate(t.values().begin() + q * 3, t.values().begin() + (q + 1) * 3, 0.0), 1, 1e-14);
            for (unsigned k = 0; k < 2; ++k) {
                double x = 0;
                for (unsigned a = 0; a < 3; ++a) {
                    x += t.values()[q * 3 + a] * m.nodes_[a].position_[k];
                }
                EXPECT_NEAR(t.points()[q][k], x, 1e-14);
            }
        }
    }
    TEST(Geometry, ZeroLeadingPivotIsInvertible) {
        auto m = data();
        std::swap(m.nodes_[1].position_, m.nodes_[2].position_);
        auto p = geometry(m);
        EXPECT_EQ(p->cells()[0].orientation(), -1);
        EXPECT_DOUBLE_EQ(p->cells()[0].measure(), 0.5);
    }
    TEST(Geometry, ScaleAppliedOnceAndPlanarityInMeters) {
        auto m = data();
        for (auto& n : m.nodes_) {
            for (auto& x : n.position_) {
                x *= 1000;
            }
            n.position_[2] = 1e-10;
        }
        GeometryOptions o;
        o.coordinateScale_ = 1e-3;
        auto p = geometry(m, o);
        EXPECT_DOUBLE_EQ(p->cells()[0].measure(), 0.5);
        for (auto x : p->cells()[0].vertices()) {
            EXPECT_EQ(x[2], 0);
        }
        m.nodes_[0].position_[2] = 1e-5;
        EXPECT_EQ(error(m, o), EvaluationCode::NonplanarCell);
    }
    TEST(Geometry, SmallWellShapedCellsRemainValid) {
        auto m = data();
        for (auto& n : m.nodes_) {
            for (auto& x : n.position_) {
                x *= 1e-100;
            }
        }
        auto p = geometry(m);
        EXPECT_NEAR(p->cells()[0].measure() / 1e-200, 0.5, 1e-14);
    }
    TEST(Geometry, DegenerateAndUnrepresentableMeasureFail) {
        auto m = data();
        m.nodes_[2].position_ = {2, 0, 0};
        EXPECT_EQ(error(m), EvaluationCode::DegenerateCell);
        m = data();
        for (auto& n : m.nodes_) {
            for (auto& x : n.position_) {
                x *= 1e-200;
            }
        }
        EXPECT_EQ(error(m), EvaluationCode::NumericalUnderflow);
    }
    TEST(Geometry, ConditionThresholdAndOptionsAreEnforced) {
        auto m = data();
        m.nodes_[2].position_[1] = 1e-5;
        GeometryOptions o;
        o.maxCondition_ = 100;
        EXPECT_EQ(error(m, o), EvaluationCode::IllConditionedCell);
        o.maxCondition_ = 1e7;
        EXPECT_NO_THROW(geometry(m, o));
        o.detTolerance_ = 0;
        EXPECT_THROW(geometry(m, o), std::invalid_argument);
        EXPECT_THROW(buildPreparedGeometry({}, {}), std::invalid_argument);
    }
    TEST(Geometry, CoordinateScalingOverflowFails) {
        auto m = data();
        m.nodes_[1].position_[0] = std::numeric_limits<double>::max();
        GeometryOptions o;
        o.coordinateScale_ = 2;
        EXPECT_EQ(error(m, o), EvaluationCode::CoordinateRange);
    }
    TEST(Geometry, EveryFacetNormalPointsOutwardAndTraceOrderIsPreserved) {
        for (unsigned d : {2U, 3U}) {
            auto m = data(d);
            std::swap(m.elements_[0].nodes_[0], m.elements_[0].nodes_[1]);
            GeometryOptions o;
            if (d == 3) {
                o.frame_ = GeometryFrame::SolidXYZ;
            }
            auto p = geometry(m, o);
            const auto& g = p->cells()[0];
            for (std::size_t i = 0; i < p->topology().facets_.size(); ++i) {
                auto t = evaluateFacetIntegration(p, i);
                const auto& f = p->exteriorFacet(i);
                auto locals = f.parentLocalDofs();
                std::size_t opposite = 0;
                while (std::find(locals.begin(), locals.end(), opposite) != locals.end()) {
                    ++opposite;
                }
                double dot = 0, norm2 = 0;
                for (unsigned k = 0; k < 3; ++k) {
                    dot += f.normal()[k] * (g.vertices()[opposite][k] - f.vertices()[0][k]);
                    norm2 += f.normal()[k] * f.normal()[k];
                }
                EXPECT_LT(dot, 0);
                EXPECT_NEAR(norm2, 1, 1e-14);
                auto ids = facetDofs(*p->space(), f.use());
                for (std::size_t a = 0; a < locals.size(); ++a) {
                    EXPECT_EQ(ids[a], p->space()->cellDofs(0)[locals[a]]);
                }
                double measure = std::accumulate(t.physicalWeights().begin(), t.physicalWeights().end(), 0.0);
                EXPECT_NEAR(measure, f.measure(), 1e-14);
                for (std::size_t q = 0; q < t.pointCount(); ++q) {
                    double sum = 0;
                    for (std::size_t a = 0; a < locals.size(); ++a) {
                        sum += t.traceValues()[q * locals.size() + a];
                    }
                    EXPECT_NEAR(sum, 1, 1e-14);
                    for (unsigned k = 0; k < d; ++k) {
                        double x = 0;
                        for (std::size_t a = 0; a < locals.size(); ++a) {
                            x += t.traceValues()[q * locals.size() + a] * g.vertices()[locals[a]][k];
                        }
                        EXPECT_NEAR(x, t.points()[q][k], 1e-14);
                    }
                }
            }
        }
    }
    TEST(Geometry, AdjacentCellsHaveOppositeNormalsAndInternalSideIsRejected) {
        auto m = data();
        m.nodes_.push_back({80, {1, 1, 0}, {}, {}});
        auto& b = m.elements_[0];
        b.tags_.push_back(201);
        b.offsets_.push_back(6);
        b.nodes_.insert(b.nodes_.end(), {1, 3, 2});
        auto p = geometry(m);
        bool found = false;
        for (std::size_t i = 0; i < p->topology().facets_.size(); ++i) {
            if (p->topology().facets_[i].uses_.size() == 2) {
                found = true;
                EXPECT_THROW(evaluateFacetIntegration(p, i), std::invalid_argument);
            }
        }
        EXPECT_TRUE(found);
        m.nodes_[3].position_ = {0.1, 0.1, 0};
        EXPECT_EQ(error(m), EvaluationCode::InconsistentNeighbors);
    }
    TEST(Geometry, NonmanifoldFacetFails) {
        auto m = data();
        m.nodes_.push_back({80, {1, 1, 0}, {}, {}});
        m.nodes_.push_back({81, {2, 2, 0}, {}, {}});
        auto& b = m.elements_[0];
        b.tags_ = {200, 201, 202};
        b.offsets_ = {0, 3, 6, 9};
        b.nodes_ = {0, 1, 2, 1, 3, 2, 1, 4, 2};
        EXPECT_EQ(error(m), EvaluationCode::NonmanifoldTopology);
    }
    TEST(Geometry, CountBudgetsAndIndexFailures) {
        auto s = space();
        EXPECT_THROW(buildPreparedGeometry(s, {}, 1, 2), EvaluationError);
        EXPECT_THROW(buildPreparedGeometry(s, {}, 0, 10), std::invalid_argument);
        auto p = buildPreparedGeometry(s, {});
        EXPECT_THROW(evaluateCellIntegration(p, 1), std::out_of_range);
        EXPECT_THROW(evaluateFacetIntegration(p, 99), std::out_of_range);
        EXPECT_THROW(evaluateCellIntegration({}, 0), std::invalid_argument);
    }
    /** @brief 小阶乘供独立矩oracle。@param n 小非负数。@pre n<10。
     * @return n!。@code auto f=factorial(3); @endcode */
    double factorial(unsigned n) {
        double f = 1;
        for (unsigned i = 2; i <= n; ++i) {
            f *= i;
        }
        return f;
    }
    TEST(Quadrature, AllMonomialsThroughDegreeTwoMatchAnalyticMoments) {
        for (unsigned d : {1U, 2U, 3U}) {
            auto family = d == 1 ? mesh::CellFamily::Segment
                : d == 2         ? mesh::CellFamily::Triangle
                                 : mesh::CellFamily::Tetrahedron;
            const auto& r = quadratureRule(family, 2);
            for (unsigned a = 0; a <= 2; ++a) {
                for (unsigned b = 0; b <= (d > 1 ? 2 - a : 0); ++b) {
                    for (unsigned c = 0; c <= (d > 2 ? 2 - a - b : 0); ++c) {
                        double sum = 0;
                        for (std::size_t q = 0; q < r.points().size(); ++q) {
                            const auto p = r.points()[q];
                            sum += r.weights()[q] * std::pow(p[0], a) * std::pow(p[1], b) * std::pow(p[2], c);
                            EXPECT_GT(r.weights()[q], 0);
                        }
                        EXPECT_NEAR(sum, factorial(a) * factorial(b) * factorial(c) / factorial(d + a + b + c), 1e-14);
                    }
                }
            }
        }
    }
    TEST(Quadrature, RequestsReportActualDegreeAndRejectUnsupported) {
        EXPECT_EQ(quadratureRule(mesh::CellFamily::Triangle, 0).exactDegree(), 1);
        EXPECT_EQ(quadratureRule(mesh::CellFamily::Triangle, 1).points().size(), 1);
        EXPECT_THROW(quadratureRule(mesh::CellFamily::Triangle, 3), std::invalid_argument);
        EXPECT_THROW(quadratureRule(mesh::CellFamily::Hexahedron, 2), std::invalid_argument);
    }
    TEST(Integration, OwnsTablesAfterGeometryIsReleasedAndAllowsConcurrentQueries) {
        auto p = geometry(data());
        auto t = evaluateCellIntegration(p, 0);
        auto task = std::async(std::launch::async, [p] { return evaluateCellIntegration(p, 0).points()[0]; });
        const auto expected = t.points()[0];
        p.reset();
        EXPECT_EQ(task.get(), expected);
        EXPECT_EQ(t.space()->cellCount(), 1);
    }
} // namespace
