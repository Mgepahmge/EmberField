/** @file Preconditioner.cuh
 * @brief 定义Preconditioner的独立接口与契约。
 */
#pragma once
#include <Numerics/LinearAlgebra/IterationContext.cuh>
namespace emberfield {
    /** @brief 预条件接口，apply 覆盖输出，不依赖其原值。
     * @details 表示 z=M^-1*rhs 的动作，不要求显式存储 M。预条件子可以组合平滑器、
     * 多层矩阵和粗层求解器，外层求解器只依赖此接口。
     * BiCGStab 要求同一次求解内使用固定的线性预条件动作；自适应或非线性动作
     * 不能仅因满足本接口就直接替换使用，需选择相容的外层算法。
     * @code std::unique_ptr<Preconditioner<double>> p = std::make_unique<Amg<double>>(); @endcode
     */
    template <linalg::Scalar T, CsrIndex I = int>
    class Preconditioner {
    public:
        /** @brief 多态释放资源，无参数、无额外约束。@note 无返回值。
         * @code p.reset(); @endcode
         */
        virtual ~Preconditioner() = default;
        /** @brief 建立预条件状态。@param matrix 系数矩阵。
         * @pre 非空有限值方阵。@note 无返回值。@code p.setup(matrix); @endcode
         */
        virtual void setup(const CsrMatrix<T, I>& matrix) = 0;
        /** @brief 近似求解并覆盖输出。@param rhs 输入。@param x 输出。
         * @pre setup 成功，尺寸设备匹配。@note 无返回值。@code p.apply(rhs, x); @endcode
         */
        virtual void apply(const Vector<T>& rhs, Vector<T>& x) = 0;
    };
} // namespace emberfield
