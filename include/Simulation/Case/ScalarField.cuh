/** @file ScalarField.cuh @brief 文件与设备无关的常量及分段时间标量场。 */
#pragma once
#include <array>
#include <variant>
#include <vector>

/** @brief 物理参数值及独立时间求值。 */
namespace emberfield::problem {
    /** @brief 有限常量，单位由使用位置决定。@code ConstantScalar c{300}; @endcode */
    struct ConstantScalar {
        double value_{}; ///< 常量值。
    };
    /** @brief 时间区间内的插值方式。 */
    enum class TimeInterpolation { Linear, Previous };
    /** @brief 时间表区间外策略。 */
    enum class TimeOutside { Error, Hold };
    /** @brief 拥有严格有序时间节点的分段函数。
     * @code TimeTableScalar t{{{0,300},{1,400}},TimeInterpolation::Linear}; @endcode */
    struct TimeTableScalar {
        std::vector<std::array<double, 2>> points_; ///< 绝对时间秒数与物理值，时间严格递增。
        TimeInterpolation interpolation_{TimeInterpolation::Linear}; ///< 线性或右连续保持。
        TimeOutside outside_{TimeOutside::Error}; ///< 默认禁止外推。
    };
    using ScalarField = std::variant<ConstantScalar, TimeTableScalar>; ///< 封闭场变体。
    /** @brief 验证场结构，无物理单位判断。@param field 借用场。@pre 无。
     * @note 无返回值；非法结构或非有限值抛invalid_argument。
     * @code validateScalarField(field); @endcode */
    void validateScalarField(const ScalarField& field);
    /** @brief 求值常量场。@param field 借用场。@pre 必须为常量且有限。
     * @return 标量；时间场或非法数值抛invalid_argument。@code auto k=evaluate(field); @endcode */
    [[nodiscard]] double evaluate(const ScalarField& field);
    /** @brief 在绝对物理时间求值。@param field 已验证场。@param time 有限秒数。
     * @pre field已通过validateScalarField且调用期间不变。
     * @return 有限标量；非法时间invalid_argument，区间外out_of_range。
     * @code auto value=evaluate(field,0.5); @endcode */
    [[nodiscard]] double evaluate(const ScalarField& field, double time);
    /** @brief 检查两条已验证曲线在闭区间内相等（包括节点左右极限）。
     * @param left 左场。@param right 右场。@param start 起点。@param end 终点。
     * @pre 曲线覆盖区间且start<=end。@return 相等返回true，无分配以外副作用。
     * @code bool same=equivalentFields(a,b,0,10); @endcode */
    [[nodiscard]] bool equivalentFields(const ScalarField& left, const ScalarField& right, double start, double end);
} // namespace emberfield::problem
