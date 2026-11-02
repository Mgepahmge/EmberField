/** @file Solver.cuh
 * @brief 定义Solver的独立接口与契约。
 */
#pragma once
#include <Solver/Iterative/SolveResult.cuh>
namespace emberfield {
    /** @brief 迭代求解接口，使用传入初值并返回明确的终止报告。
     * @details setup 绑定系数矩阵及配置，apply 负责迭代控制与终止报告。
     * 求解器通过组合持有预条件子，不从预条件子继承，也不依赖具体平滑器。
     * 正常返回必须区分收敛、迭代上限和数值失败；调用契约错误通过异常报告。
     * @code std::unique_ptr<Solver<double>> solver = std::make_unique<BiCgStab<double>>(); @endcode
     */
    template <linalg::Scalar T, CsrIndex I = int>
    class Solver {
    public:
        /** @brief 多态释放资源，无参数、无额外约束。@note 无返回值。
         * @code solver.reset(); @endcode
         */
        virtual ~Solver() = default;
        /** @brief 建立矩阵及预条件状态。@param matrix 系数矩阵。
         * @pre 非空有限值方阵。@note 无返回值。@code solver.setup(matrix); @endcode
         */
        virtual void setup(const CsrMatrix<T, I>& matrix) = 0;
        /** @brief 从初值求解 A*x=rhs。@param rhs 右端项。@param x 输入输出解。
         * @pre setup 成功，尺寸设备匹配。@return 终止报告。@code auto report = solver.apply(rhs, x); @endcode
         */
        virtual SolveResult<T> apply(const Vector<T>& rhs, Vector<T>& x) = 0;
    };
} // namespace emberfield
