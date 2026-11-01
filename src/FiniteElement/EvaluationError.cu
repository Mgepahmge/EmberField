/** @file EvaluationError.cu
 * @brief 实现独立数值错误的稳定代码转换。
 */
#include <FiniteElement/EvaluationError.cuh>
namespace thermofem::fem {
    const char* evaluationCodeName(EvaluationCode code) noexcept {
        switch (code) {
        case EvaluationCode::CoordinateRange:
            return "COORDINATE_RANGE";
        case EvaluationCode::NonplanarCell:
            return "NONPLANAR_CELL";
        case EvaluationCode::DegenerateCell:
            return "DEGENERATE_CELL";
        case EvaluationCode::IllConditionedCell:
            return "ILL_CONDITIONED_CELL";
        case EvaluationCode::NonmanifoldTopology:
            return "NONMANIFOLD_TOPOLOGY";
        case EvaluationCode::OrientationUnresolved:
            return "ORIENTATION_UNRESOLVED";
        case EvaluationCode::InconsistentNeighbors:
            return "INCONSISTENT_NEIGHBORS";
        case EvaluationCode::NonfiniteIntegration:
            return "NONFINITE_INTEGRATION";
        case EvaluationCode::NumericalOverflow:
            return "NUMERICAL_OVERFLOW";
        case EvaluationCode::NumericalUnderflow:
            return "NUMERICAL_UNDERFLOW";
        case EvaluationCode::PreparationLimit:
            return "PREPARATION_LIMIT";
        }
        return "EVALUATION_UNKNOWN";
    }
} // namespace thermofem::fem
