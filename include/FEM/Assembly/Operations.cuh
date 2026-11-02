/** @file Operations.cuh
 * @brief 提供设备局部矩阵向量的稳定分段散加与事务性累加器。
 */
#pragma once
#include <DataStructure/Storage/CsrMatrix.cuh>
#include <DataStructure/Storage/Vector.cuh>
#include <FEM/Assembly/Plan.cuh>
namespace thermofem::assembly {
    using Matrix = CsrMatrix<double, Index>; ///< 规范全局双精度64位CSR。
    /** @brief 接管完整组装结果的组合值。
     * @code auto values=std::move(accumulator).release(); @endcode */
    struct AssemblyValues {
        Matrix matrix_; ///< 全局矩阵所有权。
        Vector<double> load_; ///< 全局载荷所有权。
    };
    /** @brief 基于共享符号计划持有可复用设备累加状态。
     * @details 每次add先在候选中归约并检查，再统一提交；参数或数值校验失败保留旧状态。
     * CUDA执行故障后应丢弃累加器，不能依赖故障提交的完整性。
     * 不可并发修改同一实例；输入条目必须严格按指定模板切片平铺。
     * @code AssemblyAccumulator a(plan); a.addMatrix(0,2,entries); @endcode */
    class AssemblyAccumulator {
    public:
        /** @brief 分配零矩阵载荷。@param plan 共享符号计划。@pre 非空，当前CUDA设备可用。
         * @note 无返回；参数/资源错误传播。@code AssemblyAccumulator a(plan); @endcode */
        explicit AssemblyAccumulator(std::shared_ptr<const AssemblyPlan> plan);
        /** @brief 稳定散加局部方阵。@param firstStencil 首模板。@param count 模板数。
         * @param entries 设备平铺条目。@pre 非空有效切片，同设备global，长度精确匹配。
         * @note 无返回；形状错误invalid_argument，切片越界out_of_range，数值失败overflow_error。
         * @code a.addMatrix(0,2,entries); @endcode */
        void addMatrix(std::size_t firstStencil, std::size_t count, const CudaBuffer<double>& entries);
        /** @brief 稳定散加局部向量。@param firstStencil 首模板。@param count 模板数。
         * @param entries 设备平铺条目。@pre 同addMatrix，按局部向量序。
         * @note 无返回；参数或数值校验失败保留旧状态。@code a.addVector(0,2,entries); @endcode */
        void addVector(std::size_t firstStencil, std::size_t count, const CudaBuffer<double>& entries);
        /** @brief 只读矩阵，无参数无前提。@return 借用矩阵。@code auto& m=a.matrix(); @endcode */
        [[nodiscard]] const Matrix& matrix() const noexcept { return matrix_; }
        /** @brief 只读载荷，无参数无前提。@return 借用向量。@code auto& f=a.load(); @endcode */
        [[nodiscard]] const Vector<double>& load() const noexcept { return load_; }
        /** @brief 接管结果并结束累加；无参数。@pre 外部访问已同步。
         * @return 完整拥有结果；来源只可析构或赋值。@code auto v=std::move(a).release(); @endcode */
        [[nodiscard]] AssemblyValues release() && { return {std::move(matrix_), std::move(load_)}; }

    private:
        /** @brief 实施一种切片散加。@param first 首模板。@param count 数量。@param entries 输入。
         * @param matrix 是否方阵。@pre 输入合法。@note 无返回，提交前校验失败保留原结果。
         * @code add(0,1,b,true); @endcode */
        void add(std::size_t first, std::size_t count, const CudaBuffer<double>& entries, bool matrix);
        std::shared_ptr<const AssemblyPlan> plan_; ///< 共享只读符号计划。
        Matrix matrix_; ///< 当前设备矩阵。
        Vector<double> load_; ///< 当前设备载荷。
    };
} // namespace thermofem::assembly
