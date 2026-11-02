/** @file GpuPreparation.cu
 * @brief 批量适配物性和条件，所有局部积分及源向量叠加在GPU执行。
 */
#include <Simulation/Case/GpuPreparation.cuh>
#include <algorithm>
#include <cmath>
#include "ThermalSampling.cuh"
namespace thermofem::problem {
    namespace {
        /** @brief 在分配积分表前检查实体及平坦数组预算。@param count 实体数。
         * @param scalars 每实体最大数组长度。@param limits 预算。@pre 无。
         * @note 无返回值，非法参数invalid_argument，超额length_error。@code budget(n,48,limits); @endcode */
        void budget(std::size_t count, std::size_t scalars, thermal::GpuBatchLimits limits) {
            if (!count || !limits.maxElements_ || !limits.maxScalars_) {
                throw std::invalid_argument("GPU请求非空且预算为正");
            }
            if (count > limits.maxElements_ || count > limits.maxScalars_ / scalars) {
                throw std::length_error("GPU算例批超出限额");
            }
        }
    } // namespace
    GpuThermalCellTerms evaluateCellBatch(const PreparedThermalCase& p, std::span<const std::size_t> rows,
                                          thermal::GpuBatchLimits limits) {
        const auto d = p.space()->dimension();
        budget(rows.size(), d == 2 ? 18 : 48, limits);
        std::vector<fem::CellIntegrationData> tables;
        tables.reserve(rows.size());
        for (auto row : rows) {
            tables.push_back(fem::evaluateCellIntegration(p.geometry(), row, 2));
        }
        auto batch = thermal::uploadCellBatch(tables, limits);
        const auto q = batch.pointCount();
        const auto count = batch.size();
        const auto& def = p.bound()->data().definition_;
        const auto& c = *p.conditions();
        std::vector<double> k(count * q), capacitySamples(count * q), sources(count * q, 0),
            scales(count, detail::thermalSectionScale(p));
        std::size_t maxSources = 0;
        for (std::size_t e = 0; e < count; ++e) {
            const auto row = rows[e];
            const auto& material = def.materials_.at(c.materialIndices().at(row));
            std::fill_n(k.begin() + e * q, q, evaluate(material.conductivity_));
            if (def.analysis_.type_ == AnalysisType::Transient) {
                std::fill_n(capacitySamples.begin() + e * q, q,
                            detail::thermalCapacity(material, p.space()->cells()[row]));
            }
            maxSources = std::max(maxSources, c.sourceOffsets().at(row + 1) - c.sourceOffsets().at(row));
        }
        auto conductivity = thermal::integrateConductivityBatch(batch, k, scales);
        std::optional<thermal::GpuLocalMatrixBatch> mass;
        if (def.analysis_.type_ == AnalysisType::Transient) {
            mass = thermal::integrateCapacityBatch(batch, capacitySamples, scales);
        }
        // 源按每单元的定义序逐项积分后在GPU相加，不先把Q相加以改变数值范围。
        auto load = thermal::integrateVolumeSourceBatch(batch, sources, scales);
        for (std::size_t slot = 0; slot < maxSources; ++slot) {
            std::fill(sources.begin(), sources.end(), 0);
            for (std::size_t e = 0; e < count; ++e) {
                const auto begin = c.sourceOffsets()[rows[e]], end = c.sourceOffsets()[rows[e] + 1];
                if (slot < end - begin) {
                    const auto& source = def.sources_.at(c.sourceIndices()[begin + slot]);
                    std::fill_n(sources.begin() + e * q, q, evaluate(source.value_));
                }
            }
            auto contribution = thermal::integrateVolumeSourceBatch(batch, sources, scales);
            load = thermal::addLocalVectorBatches(batch, load, contribution);
        }
        return GpuThermalCellTerms({rows.begin(), rows.end()}, std::move(conductivity), std::move(mass),
                                   std::move(load));
    }
    GpuThermalBoundaryTerms evaluateBoundaryBatch(const PreparedThermalCase& p, std::span<const std::size_t> rows,
                                                  thermal::GpuBatchLimits limits) {
        budget(rows.size(), p.space()->dimension() == 2 ? 4 : 9, limits);
        const auto& c = *p.conditions();
        const auto& def = p.bound()->data().definition_;
        const auto& first = c.naturalTerms().at(rows[0]);
        const bool flux = std::holds_alternative<FluxBoundary>(def.boundaries_.at(first.definitionIndex()).value_);
        std::vector<fem::FacetIntegrationData> tables;
        tables.reserve(rows.size());
        for (auto row : rows) {
            const auto& term = c.naturalTerms().at(row);
            if (std::holds_alternative<FluxBoundary>(def.boundaries_.at(term.definitionIndex()).value_) != flux) {
                throw std::invalid_argument("同一个GPU边界批必须为相同自然条件类型");
            }
            auto table = fem::evaluateFacetIntegration(p.geometry(), term.facetIndex(), 2);
            if (table.use().cell_ != term.use().cell_ || table.use().localSide_ != term.use().localSide_ ||
                table.parentLocalDofs() != term.parentLocalDofs()) {
                throw std::logic_error("GPU侧关联不匹配");
            }
            tables.push_back(std::move(table));
        }
        auto batch = thermal::uploadFacetBatch(tables, limits);
        const auto q = batch.pointCount();
        std::vector<double> samples(rows.size() * q), ambient(rows.size() * q),
            scales(rows.size(), detail::thermalSectionScale(p));
        for (std::size_t e = 0; e < rows.size(); ++e) {
            const auto& term = c.naturalTerms()[rows[e]];
            const auto& value = def.boundaries_[term.definitionIndex()].value_;
            if (flux) {
                std::fill_n(samples.begin() + e * q, q, evaluate(std::get<FluxBoundary>(value).flux_));
            }
            else {
                const auto& convection = std::get<ConvectionBoundary>(value);
                std::fill_n(samples.begin() + e * q, q, evaluate(convection.coefficient_));
                std::fill_n(ambient.begin() + e * q, q, evaluate(convection.ambientTemperature_));
            }
        }
        if (flux) {
            return GpuThermalBoundaryTerms({rows.begin(), rows.end()},
                                           thermal::integrateOutwardFluxBatch(batch, samples, scales));
        }
        return GpuThermalBoundaryTerms({rows.begin(), rows.end()},
                                       thermal::integrateConvectionBatch(batch, samples, ambient, scales));
    }
} // namespace thermofem::problem
