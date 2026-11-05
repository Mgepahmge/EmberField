/** @file Archive.cuh @brief 多文件结果会话、发布游标和运行记录。 */
#pragma once
#include <Simulation/Case/Postprocess.cuh>
#include <Solver/Iterative/SolveResult.cuh>
namespace emberfield::io {
    /** @brief 本步求解报告，与温度快照解耦。@code StepRecord r; @endcode */
    struct StepRecord {
        std::uint64_t step_{}; ///< 初态/稳态0，推进从1。
        std::optional<double> time_, dt_; ///< 物理秒数。
        std::string kind_{"accepted"}, status_{"converged"}, errorCode_; ///< 机器状态。
        std::optional<SolveResult<double>> solve_; ///< 求解器原始报告，不丢失数值失败。
        std::optional<double> acceptedResidual_; ///< 投影后真实残差。
        double setupS_{}, solveS_{}, postprocessS_{}; ///< 非重叠阶段秒数。
    };
    /** @brief 显式管理独占输出目录和多文件提交前缀。
     * @details 默认拒绝非空目录；显式overwrite仅备份带有效本项目摘要的旧目录。
     * 析构仅释放锁，不隐式终结运行；必须调用finish。非并发，不可复制。
     * @code ResultArchive a(directory,input,model,plan); a.finish(true); @endcode */
    class ResultArchive {
    public:
        /** @brief 获取独占会话并发布running摘要。@param directory 最终目录。
         * @param input 配置来源。@param model 同源模型。@param plan 后处理计划。
         * @pre 模型和计划有效，目录不包含输入或网格。@note 无返回值。
         * @throws std::invalid_argument 不安全路径；filesystem_error或ios_base::failure文件错误。
         * @code ResultArchive a(path,input,model,plan); @endcode */
        ResultArchive(const std::filesystem::path& directory, const std::filesystem::path& input,
                      const problem::AssembledThermalModel& model, const problem::ThermalPostprocessPlan& plan);
        ~ResultArchive() noexcept; ///< 只释放独占锁，不抛异常。
        ResultArchive(const ResultArchive&) = delete; ///< 不复制会话。
        ResultArchive& operator=(const ResultArchive&) = delete; ///< 不复制会话。
        /** @brief 标记数值接受，独立于持久化。@param step 步。@param time 时间。
         * @pre 顺序调用。@note 无返回值。@code a.accepted(1,0.1); @endcode */
        void accepted(std::uint64_t step, std::optional<double> time);
        /** @brief 发布完整一状态。@param record 求解记录。@param fields 后处理，失败尝试为空。
         * @param saveFrame 调度结果。@pre fields与record同状态，不可在finish后调用。
         * @note 无返回值；文件异常传播，旧摘要前缀仍为权威。
         * @code a.publishStep(record,&fields,true); @endcode */
        void publishStep(const StepRecord& record, const problem::ThermalFields* fields, bool saveFrame);
        /** @brief 记录已完成阶段耗时。@param stage 协议阶段。@param seconds 非负有限秒数。
         * @pre 无并发。@note 无返回值；非法invalid_argument。@code a.timing("input",.1); @endcode */
        void timing(const std::string& stage, double seconds);
        /** @brief 记录预检时间调度的计数。@param eventCount 去重内部节点数，不含起止。
         * @param plannedStepCount 预计实际步数。@pre 瞬态会话未结束，计数来自同源游标。
         * @note 无返回值；非法会话logic_error，IO异常传播。
         * @code a.timeEvaluation(3,100); @endcode */
        void timeEvaluation(std::size_t eventCount, std::size_t plannedStepCount);
        /** @brief 追加已有输入/准备诊断，保留警告计数。@param diagnostic 已限制长度的诊断。
         * @param stage 来源阶段。@pre 会话有效。@note 无返回值；IO错误传播。
         * @code a.addDiagnostic(d,"preparation"); @endcode */
        void addDiagnostic(const problem::Diagnostic& diagnostic, const std::string& stage);
        /** @brief 显式终结。@param success 求解是否成功。@param code 失败代码。
         * @param message 中文解释。@param outputFailure 是否输出失败。
         * @param interrupted 是否用户请求的安全中断。
         * @pre 会话未终结。@note 无返回值，IO失败传播；析构不重试。
         * @code a.finish(false,"SOLVER_FAILED","未收敛"); @endcode */
        void finish(bool success, std::string code = {}, std::string message = {}, bool outputFailure = false,
                    bool interrupted = false);

    private:
        struct Impl; ///< 模块内部清单、表格前缀和锁。
        std::unique_ptr<Impl> impl_; ///< 独占会话。
    };
    /** @brief 无状态输出调度。@param options 已验证选项。@param transient 分析类型。
     * @param step 步。@param final 终态。@param initial 初态。
     * @pre 间隔非零。@return 是否保存VTU。@code auto save=shouldSave(o,true,1,false,false); @endcode */
    bool shouldSave(const problem::OutputOptions& options, bool transient, std::uint64_t step, bool final,
                    bool initial);
} // namespace emberfield::io
