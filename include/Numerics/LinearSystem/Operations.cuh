/** @file Operations.cuh
 * @brief 独立构造稳态、后向Euler系统及保留全维数的对称规定值消元。
 */
#pragma once
#include <FEM/Assembly/Operations.cuh>
/** @brief 与网格、材料和求解器解耦的设备线性系统运算。 */
namespace emberfield::systems {
    /** @brief 将规定温度精确写入温度场副本，供迭代解转为下一时刻物理状态。
     * @param state 全维候选温度。@param dofs 严格递增编号。@param values 非负有限规定值。
     * @pre 候选温度有限非负且为当前设备global。@return 独立副本，输入不变。
     * @throws std::invalid_argument 形状、编号或温度错误；资源/CUDA异常传播。
     * @code previous=projectTemperatureState(solution,dofs,g); @endcode */
    [[nodiscard]] Vector<double> projectTemperatureState(const Vector<double>& state,
                                                         std::span<const std::uint64_t> dofs,
                                                         std::span<const double> values);
    /** @brief 独立检查全温度场及常值规定温度。@param state 完整绝对温度。
     * @param dofs 严格递增约束编号。@param values 等长非负有限值。
     * @pre 当前设备global，非空，温度有限非负；约束值必须精确匹配。
     * @note 无返回；非法参数invalid_argument，CUDA异常传播。
     * @code validateTemperatureState(previous,dofs,g); @endcode */
    void validateTemperatureState(const Vector<double>& state, std::span<const std::uint64_t> dofs,
                                  std::span<const double> values);
    using Matrix = assembly::Matrix; ///< 64位规范CSR。
    /** @brief 无复制校验空间矩阵及载荷。@param spatial 非空方阵。@param load 等维向量。
     * @pre 当前设备global。@note 无返回；形状、CSR或非有限值抛invalid_argument，CUDA异常传播。
     * @code validateSystemValues(s,f); @endcode */
    void validateSystemValues(const Matrix& spatial, const Vector<double>& load);
    class LinearSystem;
    namespace detail {
        class SystemAccess;
    }
    /** @brief 组合拥有最终矩阵与右端项，不混入求解器或初猜。
     * @details 工厂验证有限性和规范结构，输出global存储；复制独立拥有设备数据。
     * @code solver.setup(system.matrix()); auto result=solver.apply(system.rhs(),x); @endcode */
    class LinearSystem {
    public:
        /** @brief 矩阵，无参数无前提。@return 生命周期内只读引用。@code auto& a=s.matrix(); @endcode */
        [[nodiscard]] const Matrix& matrix() const noexcept { return matrix_; }
        /** @brief 右端项，无参数无前提。@return 生命周期内只读引用。@code auto& b=s.rhs(); @endcode */
        [[nodiscard]] const Vector<double>& rhs() const noexcept { return rhs_; }

    private:
        friend class detail::SystemAccess;
        /** @brief 接管完整候选。@param matrix 矩阵。@param rhs 右端。@pre 同维同设备且有效。
         * @note 无返回。@code LinearSystem s(std::move(a),std::move(b)); @endcode */
        LinearSystem(Matrix matrix, Vector<double> rhs) : matrix_(std::move(matrix)), rhs_(std::move(rhs)) {}
        Matrix matrix_; ///< 拥有矩阵及完整稀疏图。
        Vector<double> rhs_; ///< 拥有全维度右端项。
    };
    /** @brief 独立复制稳态方程。@param spatial 空间项S。@param load 全载荷。
     * @pre 当前设备global，非空方阵，CSR行内排序唯一且每行有对角，输入有限。
     * @return A=S,b=f；参数错误invalid_argument，资源异常传播。
     * @code auto s=buildSteadySystem(spatial,load); @endcode */
    [[nodiscard]] LinearSystem buildSteadySystem(const Matrix& spatial, const Vector<double>& load);
    /** @brief 构造A=S+M/dt及b=f+(M/dt)*previous。@param spatial 空间项。
     * @param capacity 一致容量。@param load 载荷。@param previous 完整上一时刻场。@param timeStep 实际秒数。
     * @pre 同规范模式、设备与维数，输入有限，timeStep正有限。
     * @return 新拥有系统；非法参数invalid_argument，数值范围失败overflow_error，资源异常传播。
     * @code auto s=buildBackwardEulerSystem(k,m,f,previous,dt); @endcode */
    [[nodiscard]] LinearSystem buildBackwardEulerSystem(const Matrix& spatial, const Matrix& capacity,
                                                        const Vector<double>& load, const Vector<double>& previous,
                                                        double timeStep);
    /** @brief 同时清行清列并修正自由右端，保留全N和CSR图。
     * @param system 未约束系统。@param dofs 严格递增的全局规定值编号。
     * @param values 等长有限规定值。@pre 编号合法，当前设备匹配。
     * @return 新系统；空约束为副本，全约束为单位阵；输入不变。
     * @throws std::invalid_argument 约束错误；数值失败overflow_error，资源异常传播。
     * @code auto final=applyDirichlet(system,dofs,values); @endcode */
    [[nodiscard]] LinearSystem applyDirichlet(const LinearSystem& system, std::span<const std::uint64_t> dofs,
                                              std::span<const double> values);
} // namespace emberfield::systems
