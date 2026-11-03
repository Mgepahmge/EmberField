/** @file ResourceLimits.cuh @brief 完整运行的分阶段资源策略及逐字段输出覆盖。 */
#pragma once
#include <FEM/Space/Space.cuh>
#include <IO/Mesh/Msh.cuh>
#include <Simulation/Case/Assembly.cuh>
#include <Simulation/Case/PreparationOptions.cuh>
#include <optional>

/** @brief 完整算例运行的组合策略，不改变各计算组件的独立接口。 */
namespace emberfield::problem {
    /** @brief 调用者对算例输出预算的逐字段覆盖；空值保留算例值。
     * @details 不覆盖批大小和物理输出设置；不表示进程总内存限制。
     * @code OutputLimitOverrides o; o.maxDeviceBytes_=2147483648ULL; o.applyTo(caseOutput.limits_); @endcode */
    struct OutputLimitOverrides {
        std::optional<std::size_t> maxPoints_; ///< 活动输出顶点上限。
        std::optional<std::size_t> maxCells_; ///< 活动输出单元上限。
        std::optional<std::size_t> maxHostBytes_; ///< 后处理主机缓存字节上限。
        std::optional<std::size_t> maxDeviceBytes_; ///< 后处理设备缓存字节上限。
        std::optional<std::size_t> maxFrames_; ///< 保存帧数上限。
        std::optional<std::size_t> maxMonitors_; ///< 监测条目上限。
        std::optional<std::uint64_t> maxOutputBytes_; ///< 输出文件总字节上限。
        /** @brief 仅替换显式字段。@param limits 目标输出预算。
         * @pre 已提供的值为正。@note 无返回值、不抛异常；目标在原位更新。
         * @code overrides.applyTo(output.limits_); @endcode */
        void applyTo(OutputLimits& limits) const noexcept {
            limits.maxPoints_ = maxPoints_.value_or(limits.maxPoints_);
            limits.maxCells_ = maxCells_.value_or(limits.maxCells_);
            limits.maxHostBytes_ = maxHostBytes_.value_or(limits.maxHostBytes_);
            limits.maxDeviceBytes_ = maxDeviceBytes_.value_or(limits.maxDeviceBytes_);
            limits.maxFrames_ = maxFrames_.value_or(limits.maxFrames_);
            limits.maxMonitors_ = maxMonitors_.value_or(limits.maxMonitors_);
            limits.maxOutputBytes_ = maxOutputBytes_.value_or(limits.maxOutputBytes_);
        }
    };
    /** @brief 组合各模块原生预算，供正式运行入口统一传递。
     * @details 默认构造保持各模块默认值。数值容差和批大小不属于本文件协议。
     * 各预算独立约束对应阶段，不能保证整个进程的主机内存、显存或磁盘占用。
     * @code ResourceLimits limits; limits.case_.maxValues_=5000000; @endcode */
    struct ResourceLimits {
        CaseLimits case_; ///< 算例读取与绑定预算。
        io::MshLimits mesh_; ///< MSH输入预算。
        fem::SpaceLimits space_; ///< 有限元空间规模预算。
        PreparationOptions preparation_; ///< 几何及条件准备预算，保留原生数值默认值。
        ThermalAssemblyOptions assembly_; ///< 符号组装与GPU积分批预算，保留原生批大小。
        OutputLimitOverrides output_; ///< 验证算例之前应用的输出预算覆盖。
    };
} // namespace emberfield::problem
