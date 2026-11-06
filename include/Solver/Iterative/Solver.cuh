/** @file Solver.cuh
 * @brief 定义Solver的独立接口与契约。
 */
#pragma once
#include <Solver/Iterative/IterationObserver.cuh>
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
        Solver() = default; ///< 构造无观察者接口，无参数无前提无返回。
        Solver(const Solver&) noexcept {} ///< 复制不传播借用观察者。
        Solver(Solver&&) noexcept {} ///< 移动不传播借用观察者。
        Solver& operator=(const Solver&) noexcept { return *this; } ///< 赋值保留本实例观察者。
        Solver& operator=(Solver&&) noexcept { return *this; } ///< 移动赋值保留本实例观察者。
        /** @brief 替换借用观察者。@param observer 可空，存活覆盖订阅期间。
         * @pre 串行调用且不在apply中替换。@return 原观察者。
         * @code auto* old=solver.setObserver(&observer); @endcode */
        IterationObserver<T>* setObserver(IterationObserver<T>* observer) noexcept {
            auto* previous = observer_;
            observer_ = observer;
            return previous;
        }
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

    protected:
        /** @brief 转发已有标量。@param info 当前快照。@pre 串行、观察者有效。
         * @note 无返回，无额外计算。@code notifyIteration(info); @endcode */
        void notifyIteration(const IterationInfo<T>& info) noexcept {
            if (observer_) {
                observer_->onIteration(info);
            }
        }

    private:
        IterationObserver<T>* observer_{}; ///< 借用、不随实例复制或移动传播。
    };
    /** @brief 成对替换及恢复观察者，求解器必须比订阅存活更久。
     * @code IterationSubscription<double> subscription(solver,&observer); @endcode */
    template <linalg::Scalar T, CsrIndex I = int>
    class IterationSubscription {
    public:
        /** @brief 订阅。@param solver 借用实例。@param observer 可空观察者。
         * @pre 串行且双方生命周期覆盖本对象。@note 无返回。
         * @code IterationSubscription<double> s(solver,&observer); @endcode */
        IterationSubscription(Solver<T, I>& solver, IterationObserver<T>* observer) noexcept :
            solver_(solver), previous_(solver.setObserver(observer)) {}
        ~IterationSubscription() { solver_.setObserver(previous_); } ///< 恢复旧观察者。
        IterationSubscription(const IterationSubscription&) = delete; ///< 禁止复制。
        IterationSubscription& operator=(const IterationSubscription&) = delete; ///< 禁止赋值。
    private:
        Solver<T, I>& solver_; ///< 借用求解器。
        IterationObserver<T>* previous_; ///< 被替换的借用指针。
    };
} // namespace emberfield
