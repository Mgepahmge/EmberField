/** @file ThermalSampling.cuh
 * @brief 共享算例物性采样，不把配置解释重复嵌入主机和GPU适配器。
 */
#pragma once
#include <Simulation/Case/Preparation.cuh>
#include <cmath>
namespace thermofem::problem::detail {
    /** @brief 读取唯一厚度因子。@param p 联合对象。@pre 已准备。
     * @return 二维米制厚度或三维1。@code auto s=thermalSectionScale(p); @endcode */
    inline double thermalSectionScale(const PreparedThermalCase& p) {
        const auto& a = p.bound()->data().definition_.analysis_;
        return a.space_ == SpaceModel::Planar2d ? a.thickness_.value() : 1;
    }
    /** @brief 读取材料有效体热容。@param m 已验证材料。@param cell 单元。
     * @pre 热容形式合法。@return 正有限cV；乘积范围失败EvaluationError。
     * @code auto c=thermalCapacity(material,cell); @endcode */
    inline double thermalCapacity(const Material& m, mesh::CellRef cell) {
        const long double value = m.volumetricHeatCapacity_
            ? evaluate(*m.volumetricHeatCapacity_)
            : static_cast<long double>(evaluate(m.density_.value())) * evaluate(m.specificHeat_.value());
        const double result = static_cast<double>(value);
        if (!std::isfinite(result) || result <= 0) {
            throw fem::EvaluationError(result == 0 ? fem::EvaluationCode::NumericalUnderflow
                                                   : fem::EvaluationCode::NumericalOverflow,
                                       fem::EvaluationStage::Element, "材料 " + m.id_ + " 的体热容不可表示", cell);
        }
        return result;
    }
} // namespace thermofem::problem::detail
