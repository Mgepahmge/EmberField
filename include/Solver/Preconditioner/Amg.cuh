/** @file Amg.cuh
 * @brief 组合 GPU 原生聚合、独立平滑器、粗层求解和 V/F/W 调度的 AMG。
 */
#pragma once
#include <Solver/Preconditioner/Preconditioner.cuh>
#include <Solver/Smoother/Smoother.cuh>
#include <array>
#include <functional>
#include <memory>
#include <string>
#include <vector>
namespace thermofem {
    /** @brief AMG 循环：V 单次误差校正、W 双次粗校正、F 嵌套初值后 V 校正。 */
    enum class AmgCycle { V, W, F };
    /** @brief 配置 YHAMG-GPU 原生聚合层次和循环。
     * @code AmgOptions<double> o; o.cycle_=AmgCycle::W; @endcode */
    template <linalg::Scalar T>
    struct AmgOptions {
        int maxLevels_{20}; ///< 最大层数，1..20。
        int coarseRows_{64}; ///< 目标粗阶数，1..64。
        int matchingRounds_{8}; ///< 每阶段同步匹配轮数，1..8。
        int preSweeps_{1}, postSweeps_{1}; ///< 前后平滑次数，可为零。
        int coarseSweeps_{8}; ///< LU 不可用时的固定 Jacobi 次数。
        T jacobiWeight_{T(0.75)}; ///< 默认 Jacobi 权重。
        T strengthThreshold_{T(0.25)}; ///< 相对行最大非对角阈值。
        bool absorbSingletons_{true}; ///< 两轮单点吸收。
        bool compactSortKeys_{false}; ///< 使用紧凑坐标键。
        AmgCycle cycle_{AmgCycle::V}; ///< 循环类型。
    };
    /** @brief 每层规模和聚合统计。@code auto n=amg.levelStatistics()[0].rows_; @endcode */
    struct AmgLevelStatistics {
        std::size_t rows_{}, nonzeros_{}; ///< 本层维度与非零数。
        std::array<int, 4> aggregateSizes_{}; ///< 一到四成员聚合数。
    };
    /** @brief 组合独立 GPU 运算的非平滑聚合 AMG 预条件子。
     * @tparam T float/double。@tparam I 32/64 位索引。
     * @par 算法与组件
     * 使用用户非公开 YHAMG-GPU 原生路径：GPU 双向强图、两阶段同步互选匹配与图收缩、
     * 两轮单点吸收、分片常数 P、直接归并 Ac=P^T*A*P。每聚合最多四点，粗矩阵非零数
     * 不增加。不使用平滑延拓、截断、SpGEMM 或 cuSOLVER。AMG 源文件内的独立运算负责
     * setup、传输、粗解与循环调度；内部类型不会通过安装头文件公开。Amg 组合这些元件与
     * 每层独立 Smoother，不继承存储类型。
     * @par 粗层和停止
     * 所有层验证有限数值、非零对角和有限倒数。目标阶数、最大层数、空强图或不能收缩
     * 均可终止。粗层 <=64 阶使用 GPU 部分主元 LU，阈值与参考一致；奇异 LU 或较大
     * 终止层采用固定 coarseSweeps 次 Jacobi，保持参考的 fallback 行为。
     * @par 循环与初值
     * V 单次误差递归；W 对同一粗右端迭代两次且第二次保留第一次解；F 先限制原右端
     * 递归生成初值、延拓覆盖、再作 V 修正，与参考 F 语义一致。apply 从零开始，允许
     * rhs/x 别名，保持输出策略和地址。固定层次与固定次数构成线性近似逆动作。
     * @par 生命周期与执行
     * setup 完整构建临时状态，成功才替换，失败保留旧状态。apply 复用缓存，不可并发。
     * 原生 setup 使用 double/int；float/宽索引在 GPU 边界转换，结构受 int32 容量限制。
     * 迭代计算使用 T/I。融合内核与 CUDA Graph 未绑定到组件，因此不承诺参考原版性能；
     * 此处保留完整数值阶段，并用解耦运算支持三种调度。
     * @code AmgOptions<double> o; o.cycle_=AmgCycle::F; Amg<double> amg(o); amg.setup(a); amg.apply(b,z);
     * @endcode */
    template <linalg::Scalar T, CsrIndex I = int>
    class Amg final : public Preconditioner<T, I> {
    public:
        using SmootherFactory = std::function<std::unique_ptr<Smoother<T, I>>()>; ///< 独立组件工厂。
        /** @brief 保存配置及可选工厂。
         * @param options 配置。@param factory 分别为前后平滑创建组件。
         * @pre 配置合法；注入平滑器提供固定线性动作。@note 无返回值。
         * @details 默认使用独立 Jacobi，扫描数取 pre/postSweeps；注入工厂时扫描数由
         * 返回组件定义。对应 sweeps 为零仍跳过阶段。粗层 fallback 不使用工厂。
         * @code Amg<double> amg(options); @endcode */
        explicit Amg(AmgOptions<T> options = {}, SmootherFactory factory = {});
        /** @brief 释放层次及 GPU 缓存，无参数、无返回值。@pre 无并发访问。
         * @code { Amg<double> amg; } @endcode */
        ~Amg() override;
        Amg(const Amg&) = delete; ///< 独占层次不可复制。
        Amg& operator=(const Amg&) = delete; ///< 独占层次不可复制。
        /** @brief 转移层次所有权。@param other 来源。@pre 无并发访问。@note 无返回值。
         * @code Amg<double> moved(std::move(amg)); @endcode */
        Amg(Amg&& other) noexcept;
        /** @brief 转移层次所有权。@param other 来源。@pre 无并发访问。@return 当前对象。
         * @code moved = std::move(amg); @endcode */
        Amg& operator=(Amg&& other) noexcept;
        /** @brief 构建层次、因子、组件和缓存，失败保留旧状态。
         * @param matrix 输入方阵。@pre 非空有限，各层对角有效，原生图容量可表示。
         * @note 无返回值。@throws std::invalid_argument 非法数值、对角或空工厂。
         * @throws std::length_error 原生图容量不足。@code amg.setup(a); @endcode */
        void setup(const CsrMatrix<T, I>& matrix) override;
        /** @brief 从零执行配置的 V/W/F 循环，覆盖输出。
         * @param rhs 右端。@param x 输出，可与 rhs 别名。
         * @pre setup 成功，尺寸设备匹配，支持全部内存策略。@note 无返回值。
         * @details 先保存 rhs；调度逐层调用独立平滑、残差、限制、延拓及粗层求解接口。
         * 不判断收敛容差，调用方求解器负责真实残差验证。@code amg.apply(b,z); @endcode */
        void apply(const Vector<T>& rhs, Vector<T>& x) override;
        /** @brief 查询层数，无参数、无约束。@return 层数。@code amg.levelCount(); @endcode */
        std::size_t levelCount() const noexcept;
        /** @brief 查询层统计，无参数。@pre setup 成功。@return 只读统计。@code amg.levelStatistics(); @endcode */
        const std::vector<AmgLevelStatistics>& levelStatistics() const;
        /** @brief 查询停止原因，无参数。@pre setup 成功。@return 描述。@code amg.stopReason(); @endcode */
        const std::string& stopReason() const;
        /** @brief 查询粗层路径，无参数。@pre setup 成功。@return 描述。@code amg.coarseSolver(); @endcode */
        const std::string& coarseSolver() const;
        /** @brief 查询最近 apply 的每层访问数，无参数。@pre setup 成功。
         * @return 计数。@code amg.lastCycleVisits(); @endcode */
        const std::vector<std::size_t>& lastCycleVisits() const;

    private:
        struct Level; ///< 源文件内定义的单层状态。
        struct State; ///< 源文件内定义的事务性层次状态。
        struct CycleAdapter; ///< 源文件内定义的循环适配器。
        /** @brief 创建并建立平滑器。@param a 矩阵。@param sweeps 默认次数。
         * @pre 当前设备正确。@return 独立组件。@code auto s=makeSmoother(a,1); @endcode */
        std::unique_ptr<Smoother<T, I>> makeSmoother(const CsrMatrix<T, I>& a, int sweeps);
        /** @brief 获取已建立状态，无参数。@pre setup 成功。@return 状态。@code state(); @endcode */
        const State& state() const;
        AmgOptions<T> options_; ///< 固定配置。
        SmootherFactory factory_; ///< 可注入平滑器工厂。
        std::unique_ptr<State> state_; ///< 最近成功状态。
    };
} // namespace thermofem
