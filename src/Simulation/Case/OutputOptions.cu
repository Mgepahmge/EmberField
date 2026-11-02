/** @file OutputOptions.cu @brief 类型化输出值与文件协议名字的独立映射。 */
#include <Simulation/Case/Definition.cuh>
#include <stdexcept>
namespace emberfield::problem {
    std::string_view outputFieldName(OutputField value) {
        switch (value) {
        case OutputField::Temperature:
            return "temperature";
        case OutputField::TemperatureGradient:
            return "temperatureGradient";
        case OutputField::HeatFlux:
            return "heatFlux";
        }
        throw std::invalid_argument("未知物理场枚举");
    }
    std::string_view regionStatisticName(RegionStatistic value) {
        switch (value) {
        case RegionStatistic::TemperatureMin:
            return "temperatureMin";
        case RegionStatistic::TemperatureMax:
            return "temperatureMax";
        case RegionStatistic::TemperatureMean:
            return "temperatureMean";
        case RegionStatistic::StoredEnergy:
            return "storedEnergy";
        }
        throw std::invalid_argument("未知区域统计枚举");
    }
} // namespace emberfield::problem
