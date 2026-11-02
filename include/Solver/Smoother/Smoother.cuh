/** @file Smoother.cuh
 * @brief 定义Smoother的独立接口与契约。
 */
#pragma once
#include <Numerics/LinearAlgebra/IterationContext.cuh>
namespace emberfield {
    /** @brief 迭代平滑器接口，apply 从传入初值更新近似解。
     * @details 继承仅表达算法角色；矩阵、工作区等资源由具体实现组合持有。
     * setup 建立与矩阵关联的状态，apply 修改近似解但不改变右端项。
     * 平滑器不判断全局收敛，也不拥有外层 AMG；扫描参数由具体实现配置。
     * @note 实现必须说明初值、别名、设备、同步及 setup 失败后的状态约定。
     * @code std::unique_ptr<Smoother<double>> smoother = std::make_unique<Jacobi<double>>(); @endcode
     */
    template <linalg::Scalar T, CsrIndex I = int>
    class Smoother {
    public:
        /** @brief 多态释放资源，无参数、无额外约束。@note 无返回值。
         * @code smoother.reset(); @endcode
         */
        virtual ~Smoother() = default;
        /** @brief 保存矩阵并建立缓存。@param matrix 系数矩阵。
         * @pre 非空有限值方阵。@note 无返回值。@code smoother.setup(matrix); @endcode
         */
        virtual void setup(const CsrMatrix<T, I>& matrix) = 0;
        /** @brief 对初值执行平滑。@param rhs 右端项。@param x 输入输出近似解。
         * @pre setup 成功，尺寸设备匹配。@note 无返回值。@code smoother.apply(rhs, x); @endcode
         */
        virtual void apply(const Vector<T>& rhs, Vector<T>& x) = 0;
    };
} // namespace emberfield
