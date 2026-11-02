/** @file EvaluationError.cuh
 * @brief 定义几何与积分的稳定数值错误，不依赖物理算例。
 */
#pragma once
#include <DataStructure/Mesh/FacetTopology.cuh>
#include <stdexcept>

namespace emberfield::fem {
    /** @brief 可解释的几何和数值失败种类。 */
    enum class EvaluationCode {
        CoordinateRange,
        NonplanarCell,
        DegenerateCell,
        IllConditionedCell,
        NonmanifoldTopology,
        OrientationUnresolved,
        InconsistentNeighbors,
        NonfiniteIntegration,
        NumericalOverflow,
        NumericalUnderflow,
        PreparationLimit
    };
    /** @brief 出错的独立计算阶段。 */
    enum class EvaluationStage { Geometry, Integration, Element };
    /** @brief 获取稳定错误码。@param code 错误种类。@pre 无。
     * @return 静态字符串，无分配。@code auto s=evaluationCodeName(code); @endcode */
    [[nodiscard]] const char* evaluationCodeName(EvaluationCode code) noexcept;
    /** @brief 拥有数值失败信息，调用者不得用零贡献掩盖失败。
     * @code throw EvaluationError(EvaluationCode::DegenerateCell,EvaluationStage::Geometry,"退化单元",cell);
     * @endcode */
    class EvaluationError : public std::runtime_error {
    public:
        /** @brief 构造失败信息。@param code 稳定代码。@param stage 阶段。@param message 中文说明。
         * @param cell 可选母单元。@param localSide 可选局部侧。@param quadraturePoint 可选积分点。
         * @param localDof 可选局部自由度。@pre 上下文有实际依据。
         * @note 无返回值；分配异常传播。
         * @code EvaluationError e(code,stage,"无法表示结果"); @endcode */
        EvaluationError(EvaluationCode code, EvaluationStage stage, std::string message,
                        std::optional<mesh::CellRef> cell = {}, std::optional<std::size_t> localSide = {},
                        std::optional<std::size_t> quadraturePoint = {}, std::optional<std::size_t> localDof = {}) :
            std::runtime_error(std::move(message)), code_(code), stage_(stage), cell_(cell), localSide_(localSide),
            quadraturePoint_(quadraturePoint), localDof_(localDof) {}
        /** @brief 查询稳定代码；无参数无前提。@return 代码。@code auto c=e.code(); @endcode */
        [[nodiscard]] EvaluationCode code() const noexcept { return code_; }
        /** @brief 查询阶段；无参数无前提。@return 阶段。@code auto s=e.stage(); @endcode */
        [[nodiscard]] EvaluationStage stage() const noexcept { return stage_; }
        /** @brief 查询单元；无参数无前提。@return 可选身份。@code auto c=e.cell(); @endcode */
        [[nodiscard]] auto cell() const noexcept { return cell_; }
        /** @brief 查询局部侧；无参数无前提。@return 可选序号。@code auto s=e.localSide(); @endcode */
        [[nodiscard]] auto localSide() const noexcept { return localSide_; }
        /** @brief 查询积分点；无参数无前提。@return 可选序号。@code auto q=e.quadraturePoint(); @endcode */
        [[nodiscard]] auto quadraturePoint() const noexcept { return quadraturePoint_; }
        /** @brief 查询局部DOF；无参数无前提。@return 可选序号。@code auto a=e.localDof(); @endcode */
        [[nodiscard]] auto localDof() const noexcept { return localDof_; }

    private:
        EvaluationCode code_; ///< 稳定代码。
        EvaluationStage stage_; ///< 计算阶段。
        std::optional<mesh::CellRef> cell_; ///< 母单元。
        std::optional<std::size_t> localSide_; ///< 局部侧。
        std::optional<std::size_t> quadraturePoint_; ///< 积分点。
        std::optional<std::size_t> localDof_; ///< 局部自由度。
    };
} // namespace emberfield::fem
