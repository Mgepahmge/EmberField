/** @file IterationValidation.cuh
 * @brief 定义IterationValidation的独立接口与契约。
 */
#pragma once
#include <Numerics/LinearAlgebra/IterationContext.cuh>
namespace emberfield {
    /** @brief 提供求解组件共享的矩阵和向量前置条件检查。 */
    namespace iteration {
        /** @brief 检查非空方阵。@tparam T 精度。@tparam I 索引。@param matrix 输入。
         * @pre 无额外约束。@note 无返回值。@code validateMatrix(matrix); @endcode
         */
        template <linalg::Scalar T, CsrIndex I>
        void validateMatrix(const CsrMatrix<T, I>& matrix) {
            if (!matrix.rowCount() || matrix.rowCount() != matrix.columnCount()) {
                throw std::invalid_argument("Solver requires a nonempty square matrix");
            }
        }
        /** @brief 检查向量尺寸与设备。@tparam T 精度。@param n 维度。@param device 所属设备。
         * @param rhs 右端项。@param x 解。@pre 无额外约束。@note 无返回值。
         * @code validateVectors(n, device, rhs, x); @endcode
         */
        template <linalg::Scalar T>
        void validateVectors(std::size_t n, int device, const Vector<T>& rhs, const Vector<T>& x) {
            if (rhs.size() != n || x.size() != n) {
                throw std::invalid_argument("Solver vector shape mismatch");
            }
            linalg::detail::commonDevice({device, rhs.deviceId(), x.deviceId()});
        }
    } // namespace iteration
} // namespace emberfield
