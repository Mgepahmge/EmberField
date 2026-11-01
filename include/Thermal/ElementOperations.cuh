/** @file ElementOperations.cuh
 * @brief 提供不依赖材料查找和全局矩阵的热传导局部数值积分。
 */
#pragma once
#include <FiniteElement/Integration.cuh>

namespace thermofem::thermal {
    namespace detail {
        class LocalResultAccess;
    }
    /** @brief 局部方阵拥有值，不携带全局索引或设备资源。
     * @code auto result=integrateConductivity(data,samples,scale); @endcode */
    class LocalMatrix {
    public:
        /** @brief 局部DOF数；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.size(); @endcode */
        [[nodiscard]] std::size_t size() const noexcept { return size_; }
        /** @brief 局部行主序数组；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.entries(); @endcode
         */
        [[nodiscard]] const std::vector<double>& entries() const noexcept { return entries_; }
        /** @brief 查询局部矩阵条目。@param a 行。@param b 列。@pre 无。
         * @return 数值；越界out_of_range。@code auto x=m(0,1); @endcode */
        [[nodiscard]] double operator()(std::size_t a, std::size_t b) const {
            if (a >= size_ || b >= size_) {
                throw std::out_of_range("局部矩阵索引越界");
            }
            return entries_[a * size_ + b];
        }

    private:
        friend class detail::LocalResultAccess;
        /** @brief 接管受检局部数据。@param n 维数。@param v 数组。@pre 仅内部工厂调用。
         * @note 无返回值。@code LocalMatrix x(n,std::move(v)); @endcode */
        LocalMatrix(std::size_t n, std::vector<double> v) : size_(n), entries_(std::move(v)) {}
        std::size_t size_; ///< 局部DOF数。
        std::vector<double> entries_; ///< 拥有的有限条目。
    };
    /** @brief 局部载荷拥有值，不携带全局索引或设备资源。
     * @code auto result=integrateVolumeSource(data,samples,scale); @endcode */
    class LocalVector {
    public:
        /** @brief 局部DOF数；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.size(); @endcode */
        [[nodiscard]] std::size_t size() const noexcept { return size_; }
        /** @brief 局部行主序数组；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.entries(); @endcode
         */
        [[nodiscard]] const std::vector<double>& entries() const noexcept { return entries_; }
        /** @brief 查询局部条目。@param a 局部行。@pre 无。@return 数值；越界out_of_range。
         * @code auto x=v[0]; @endcode */
        [[nodiscard]] double operator[](std::size_t a) const { return entries_.at(a); }

    private:
        friend class detail::LocalResultAccess;
        /** @brief 接管受检局部数据。@param n 维数。@param v 数组。@pre 仅内部工厂调用。
         * @note 无返回值。@code LocalVector x(n,std::move(v)); @endcode */
        LocalVector(std::size_t n, std::vector<double> v) : size_(n), entries_(std::move(v)) {}
        std::size_t size_; ///< 局部DOF数。
        std::vector<double> entries_; ///< 拥有的有限条目。
    };
    /** @brief 同一侧对流矩阵和载荷的组合，不丢弃环境温度项。
     * @code auto c=integrateConvection(data,h,ambient,scale); @endcode */
    class ConvectionContribution {
    public:
        /** @brief 对流矩阵；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.matrix(); @endcode */
        [[nodiscard]] const LocalMatrix& matrix() const noexcept { return matrix_; }
        /** @brief 环境载荷；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.load(); @endcode */
        [[nodiscard]] const LocalVector& load() const noexcept { return load_; }

    private:
        friend class detail::LocalResultAccess;
        /** @brief 接管成对贡献。@param m 矩阵。@param v 向量。@pre 同序同尺寸。
         * @note 无返回值。@code ConvectionContribution c(std::move(m),std::move(v)); @endcode */
        ConvectionContribution(LocalMatrix m, LocalVector v) : matrix_(std::move(m)), load_(std::move(v)) {}
        LocalMatrix matrix_; ///< 迹矩阵W/K。
        LocalVector load_; ///< 环境载荷W。
    };
    /** @brief 积分导热刚度。@param data 体表。@param conductivity 按点的正有限k。
     * @param sectionScale 二维厚度或三维1。@pre 样本数等于点数，来源同表。
     * @return 局部K，W/K；非法参数invalid_argument，数值失败EvaluationError。
     * @code auto k=integrateConductivity(t,kSamples,0.01); @endcode */
    [[nodiscard]] LocalMatrix integrateConductivity(const fem::CellIntegrationData& data,
                                                    std::span<const double> conductivity, double sectionScale);
    /** @brief 积分一致容量。@param data 二次精确体表。@param volumetricCapacity 按点的正有限cV。
     * @param sectionScale 二维厚度或三维1。@pre 样本数等于点数。
     * @return 局部M，J/K，不是M/dt；非法参数invalid_argument，数值失败EvaluationError。
     * @code auto m=integrateCapacity(t,cSamples,0.01); @endcode */
    [[nodiscard]] LocalMatrix integrateCapacity(const fem::CellIntegrationData& data,
                                                std::span<const double> volumetricCapacity, double sectionScale);
    /** @brief 积分体源载荷。@param data 体表。@param source 按点的有限有符号Q。
     * @param sectionScale 二维厚度或三维1。@pre 样本数相等且规则至少一次精确。
     * @return 局部f，W；非法参数invalid_argument，数值失败EvaluationError。
     * @code auto f=integrateVolumeSource(t,qSamples,0.01); @endcode */
    [[nodiscard]] LocalVector integrateVolumeSource(const fem::CellIntegrationData& data,
                                                    std::span<const double> source, double sectionScale);
    /** @brief 积分向外通量的负载荷。@param data 侧表。@param outwardFlux 按点的q·n，正值向外。
     * @param sectionScale 二维厚度或三维1。@pre 样本数相等且规则至少一次精确。
     * @return 迹载荷，W；非法参数invalid_argument，数值失败EvaluationError。
     * @code auto f=integrateOutwardFlux(t,gSamples,0.01); @endcode */
    [[nodiscard]] LocalVector integrateOutwardFlux(const fem::FacetIntegrationData& data,
                                                   std::span<const double> outwardFlux, double sectionScale);
    /** @brief 成对积分对流项。@param data 二次精确侧表。@param coefficient 按点的有限h>=0。
     * @param ambientTemperature 按点的有限K值>=0。@param sectionScale 二维厚度或三维1。
     * @pre 样本数与点数相等且同表采样。
     * @return H及f∞；非法参数invalid_argument，数值失败EvaluationError。
     * @code auto c=integrateConvection(t,hSamples,ambientSamples,0.01); @endcode */
    [[nodiscard]] ConvectionContribution integrateConvection(const fem::FacetIntegrationData& data,
                                                             std::span<const double> coefficient,
                                                             std::span<const double> ambientTemperature,
                                                             double sectionScale);
} // namespace thermofem::thermal
