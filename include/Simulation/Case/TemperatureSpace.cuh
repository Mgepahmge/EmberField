/** @file TemperatureSpace.cuh
 * @brief 将只读绑定算例适配为有限元温度空间，不施加物理条件或更改阶段报告。
 */
#pragma once
#include <FEM/Space/Space.cuh>
#include <Simulation/Case/Binding.cuh>

namespace thermofem::problem {
    /** @brief 温度空间准备结果；成功仅证明编号完成，不代表算例可执行。
     * @code if(result.space_) inspect(result.space_->dofCount()); @endcode */
    struct TemperatureSpaceResult {
        std::shared_ptr<const fem::FiniteElementSpace> space_; ///< 未施加约束的只读温度空间。
        Diagnostics diagnostics_; ///< 算例诊断，保留底层截断错误状态。
    };
    /** @brief 将活动域与温度阶次转交独立空间构建器，不读取文件、不上传GPU。
     * @param bound 已通过身份绑定的算例。@param limits 空间资源上限，各项为正。
     * @pre 底层Mesh不得通过其他别名替换、移动或修改；调用者保证几何共形。
     * @return 空间与诊断；周期物理策略尚未定义时拒绝，不修改BoundCase。
     * @throws std::invalid_argument 非法限额；标准资源异常传播。
     * @note 不提前要求整个capability可执行，也不删除共享DOF条件尚未检查的待检项。
     * @code auto r=buildTemperatureSpace(*binding.bound_); @endcode */
    [[nodiscard]] TemperatureSpaceResult buildTemperatureSpace(const BoundCase& bound, fem::SpaceLimits limits = {});
} // namespace thermofem::problem
