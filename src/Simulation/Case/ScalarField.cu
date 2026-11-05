/** @file ScalarField.cu @brief 稳定线性插值、右连续保持与全分段曲线相容检查。 */
#include <Simulation/Case/ScalarField.cuh>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace emberfield::problem {
    void validateScalarField(const ScalarField& field) {
        if (const auto* c = std::get_if<ConstantScalar>(&field)) {
            if (!std::isfinite(c->value_)) {
                throw std::invalid_argument("常量场必须有限");
            }
            return;
        }
        const auto& t = std::get<TimeTableScalar>(field);
        if ((t.interpolation_ != TimeInterpolation::Linear && t.interpolation_ != TimeInterpolation::Previous) ||
            (t.outside_ != TimeOutside::Error && t.outside_ != TimeOutside::Hold) || t.points_.empty() ||
            (t.interpolation_ == TimeInterpolation::Linear && t.points_.size() < 2)) {
            throw std::invalid_argument("时间表类型或节点数非法");
        }
        for (std::size_t i = 0; i < t.points_.size(); ++i) {
            if (!std::isfinite(t.points_[i][0]) || !std::isfinite(t.points_[i][1]) ||
                (i && t.points_[i][0] <= t.points_[i - 1][0])) {
                throw std::invalid_argument("时间表节点必须有限且时间严格递增");
            }
        }
    }
    double evaluate(const ScalarField& field) {
        const auto* c = std::get_if<ConstantScalar>(&field);
        if (!c || !std::isfinite(c->value_)) {
            throw std::invalid_argument("无时间求值只接受有限常量场");
        }
        return c->value_;
    }
    double evaluate(const ScalarField& field, double time) {
        if (!std::isfinite(time)) {
            throw std::invalid_argument("求值时间必须有限");
        }
        if (std::holds_alternative<ConstantScalar>(field)) {
            return evaluate(field);
        }
        const auto& t = std::get<TimeTableScalar>(field);
        if (t.points_.empty()) {
            throw std::invalid_argument("时间表为空");
        }
        if (time < t.points_.front()[0] || time > t.points_.back()[0]) {
            if (t.outside_ != TimeOutside::Hold) {
                throw std::out_of_range("时间表未覆盖求值时间");
            }
            return time < t.points_.front()[0] ? t.points_.front()[1] : t.points_.back()[1];
        }
        auto upper = std::upper_bound(t.points_.begin(), t.points_.end(), time,
                                      [](double x, const auto& p) { return x < p[0]; });
        const auto& a = *(upper - 1);
        if (a[0] == time || upper == t.points_.end() || t.interpolation_ == TimeInterpolation::Previous) {
            return a[1];
        }
        const auto& b = *upper;
        // 跨越极端正负时间时先缩放，避免double差值溢出；lerp避免值差溢出。
        const double span = b[0] - a[0];
        const double alpha = std::isfinite(span) ? (time - a[0]) / span : (time / 2 - a[0] / 2) / (b[0] / 2 - a[0] / 2);
        const double value = std::lerp(a[1], b[1], std::clamp(alpha, 0.0, 1.0));
        if (!std::isfinite(value)) {
            throw std::overflow_error("时间场插值溢出");
        }
        return value;
    }
    bool equivalentFields(const ScalarField& left, const ScalarField& right, double start, double end) {
        validateScalarField(left);
        validateScalarField(right);
        if (!std::isfinite(start) || !std::isfinite(end) || start > end) {
            throw std::invalid_argument("曲线比较区间非法");
        }
        std::vector<double> points{start, end};
        for (const auto* field : {&left, &right}) {
            if (const auto* t = std::get_if<TimeTableScalar>(field)) {
                for (const auto& p : t->points_) {
                    if (p[0] > start && p[0] < end) {
                        points.push_back(p[0]);
                    }
                }
            }
        }
        std::sort(points.begin(), points.end());
        points.erase(std::unique(points.begin(), points.end()), points.end());
        const auto leftLimit = [](const ScalarField& field, double time) {
            if (const auto* table = std::get_if<TimeTableScalar>(&field);
                table && table->interpolation_ == TimeInterpolation::Previous) {
                const auto at = std::lower_bound(table->points_.begin(), table->points_.end(), time,
                                                 [](const auto& point, double t) { return point[0] < t; });
                if (at != table->points_.end() && (*at)[0] == time && at != table->points_.begin()) {
                    return (*(at - 1))[1];
                }
            }
            return evaluate(field, time);
        };
        for (double time : points) {
            if (evaluate(left, time) != evaluate(right, time)) {
                return false;
            }
            if (time > start) {
                if (leftLimit(left, time) != leftLimit(right, time)) {
                    return false;
                }
            }
        }
        return true;
    }
} // namespace emberfield::problem
