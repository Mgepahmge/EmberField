/** @file LocalResultAccess.cuh
 * @brief 内部受检局部结果构造工具，不作为公共扩展点。
 */
#pragma once
#include <Simulation/Thermal/ElementOperations.cuh>
#include <algorithm>
#include <cmath>
namespace thermofem::thermal::detail {
    /** @brief 校验局部结果后接管；主机积分和GPU回读共用。
     * @code auto m=LocalResultAccess::matrix(n,std::move(values)); @endcode */
    class LocalResultAccess {
    public:
        /** @brief 创建局部方阵。@param n 维数。@param v 有限数组。@pre n>0。
         * @return 成功值；非法形状或数值抛logic_error。@code auto m=matrix(3,v); @endcode */
        static LocalMatrix matrix(std::size_t n, std::vector<double> v) {
            if (!n || n > std::numeric_limits<std::size_t>::max() / n || v.size() != n * n ||
                !std::all_of(v.begin(), v.end(), [](double x) { return std::isfinite(x); })) {
                throw std::logic_error("内部局部矩阵形状或值非法");
            }
            return LocalMatrix(n, std::move(v));
        }
        /** @brief 创建局部载荷。@param n 维数。@param v 有限数组。@pre n>0。
         * @return 成功值；非法形状或数值抛logic_error。@code auto f=vector(3,v); @endcode */
        static LocalVector vector(std::size_t n, std::vector<double> v) {
            if (!n || v.size() != n || !std::all_of(v.begin(), v.end(), [](double x) { return std::isfinite(x); })) {
                throw std::logic_error("内部局部向量形状或值非法");
            }
            return LocalVector(n, std::move(v));
        }
        /** @brief 组合对流结果。@param m 矩阵。@param v 向量。@pre 同序同尺寸。
         * @return 成对贡献。@code auto c=convection(m,v); @endcode */
        static ConvectionContribution convection(LocalMatrix m, LocalVector v) {
            if (m.size() != v.size()) {
                throw std::logic_error("对流结果尺寸不一致");
            }
            return ConvectionContribution(std::move(m), std::move(v));
        }
    };
} // namespace thermofem::thermal::detail
