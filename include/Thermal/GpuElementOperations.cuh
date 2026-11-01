/** @file GpuElementOperations.cuh
 * @brief 批量GPU热积分，几何上传可复用，局部结果保留在设备端供后续组装。
 */
#pragma once
#include <DataStructure/CudaBuffer.cuh>
#include <Thermal/ElementOperations.cuh>
namespace thermofem::thermal {
    namespace detail {
        class GpuBatchAccess;
        struct GpuBatchIdentity;
    } // namespace detail
    /** @brief 区分体表与迹表，GPU入口严格核对kind。 */
    enum class GpuBatchKind { Cell, Facet };
    /** @brief GPU批处理形状和输入大小预算，调用方控制，不取自JSON。 */
    struct GpuBatchLimits {
        std::size_t maxElements_{65536}; ///< 一个批次的实体数量上限。
        std::size_t maxScalars_{16777216}; ///< 每个平坦数组的元素上限。
    };
    class GpuElementBatch;
    /** @brief 上传同形状体积分表。@param tables 同维、同n/q和同space的表。
     * @param limits 正预算。@pre 输入非空，空间不变。
     * @return 拥有设备几何的批；非法形状invalid_argument，超额length_error，CUDA失败CudaError。
     * @code auto b=uploadCellBatch(tables); @endcode */
    [[nodiscard]] GpuElementBatch uploadCellBatch(std::span<const fem::CellIntegrationData> tables,
                                                  GpuBatchLimits limits = {});
    /** @brief 上传同形状侧积分表。@param tables 同维、同nf/q及同space的表。
     * @param limits 正预算。@pre 非空。@return 设备迹批，失败同体批。
     * @code auto b=uploadFacetBatch(tables); @endcode */
    [[nodiscard]] GpuElementBatch uploadFacetBatch(std::span<const fem::FacetIntegrationData> tables,
                                                   GpuBatchLimits limits = {});
    /** @brief 不可变设备积分数据，与配置/材料查找解耦，可供多次独立批积分复用。
     * @details 体梯度[(e*q+r)*n+a]*d+k；值[(e*q+r)*n+a]；权重[e*q+r]。
     * 每个批次保存来源上下文，结果与来源身份绑定；无跨设备隐式复制。
     * @code auto batch=uploadCellBatch(tables); @endcode */
    class GpuElementBatch {
    public:
        /** @brief 实体数；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.size(); @endcode */
        [[nodiscard]] std::size_t size() const noexcept { return count_; }
        /** @brief 每实体DOF数；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.dofCount(); @endcode */
        [[nodiscard]] std::size_t dofCount() const noexcept { return dofs_; }
        /** @brief 每实体积分点数；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.pointCount();
         * @endcode */
        [[nodiscard]] std::size_t pointCount() const noexcept { return points_; }
        /** @brief 母元维数；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.dimension(); @endcode */
        [[nodiscard]] unsigned dimension() const noexcept { return dimension_; }
        /** @brief 实际精度；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.exactDegree(); @endcode */
        [[nodiscard]] unsigned exactDegree() const noexcept { return exactDegree_; }
        /** @brief 体或侧；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.kind(); @endcode */
        [[nodiscard]] GpuBatchKind kind() const noexcept { return kind_; }
        /** @brief 分配所属设备；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.deviceId(); @endcode
         */
        [[nodiscard]] int deviceId() const noexcept { return weights_.deviceId(); }

    private:
        GpuElementBatch() = default; ///< 上传工厂候选。
        friend class detail::GpuBatchAccess;
        std::shared_ptr<const detail::GpuBatchIdentity> identity_; ///< 来源及数值失败上下文。
        std::size_t count_{}; ///< 实体数。
        std::size_t dofs_{}; ///< 每实体局部维数。
        std::size_t points_{}; ///< 每实体积分点数。
        unsigned dimension_{}; ///< 母元维数。
        unsigned exactDegree_{}; ///< 精度。
        GpuBatchKind kind_{}; ///< 体/侧。
        CudaBuffer<double> weights_; ///< 设备实际权重。
        CudaBuffer<double> values_; ///< 设备基值或迹值。
        CudaBuffer<double> gradients_; ///< 体实际梯度，侧为空。
    };
    /** @brief 方阵批的设备所有权，结果不自动回读主机。
     * @code auto v=integrateConductivityBatch(batch,samples,scales); @endcode */
    class GpuLocalMatrixBatch {
    public:
        /** @brief 实体数；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.size(); @endcode */
        [[nodiscard]] std::size_t size() const noexcept { return count_; }
        /** @brief 每实体DOF数；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.dofCount(); @endcode */
        [[nodiscard]] std::size_t dofCount() const noexcept { return dofs_; }
        /** @brief 实体优先的设备结果；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.values();
         * @endcode */
        [[nodiscard]] const CudaBuffer<double>& values() const noexcept { return values_; }
        /** @brief 显式同步回读整个批次；无参数。@pre 无外部异步写。
         * @return 拥有的主机数组；CUDA错误传播。@code auto host=result.copyToHost(); @endcode */
        [[nodiscard]] std::vector<double> copyToHost() const;

    private:
        GpuLocalMatrixBatch() = default; ///< 仅数值工厂候选。
        friend class detail::GpuBatchAccess;
        std::shared_ptr<const detail::GpuBatchIdentity> identity_; ///< 不可混加的来源身份。
        std::size_t count_{}; ///< 实体数。
        std::size_t dofs_{}; ///< 局部维数。
        CudaBuffer<double> values_; ///< 设备局部贡献。
    };
    /** @brief 向量批的设备所有权，结果不自动回读主机。
     * @code auto v=integrateVolumeSourceBatch(batch,samples,scales); @endcode */
    class GpuLocalVectorBatch {
    public:
        /** @brief 实体数；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.size(); @endcode */
        [[nodiscard]] std::size_t size() const noexcept { return count_; }
        /** @brief 每实体DOF数；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.dofCount(); @endcode */
        [[nodiscard]] std::size_t dofCount() const noexcept { return dofs_; }
        /** @brief 实体优先的设备结果；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.values();
         * @endcode */
        [[nodiscard]] const CudaBuffer<double>& values() const noexcept { return values_; }
        /** @brief 显式同步回读整个批次；无参数。@pre 无外部异步写。
         * @return 拥有的主机数组；CUDA错误传播。@code auto host=result.copyToHost(); @endcode */
        [[nodiscard]] std::vector<double> copyToHost() const;

    private:
        GpuLocalVectorBatch() = default; ///< 仅数值工厂候选。
        friend class detail::GpuBatchAccess;
        std::shared_ptr<const detail::GpuBatchIdentity> identity_; ///< 不可混加的来源身份。
        std::size_t count_{}; ///< 实体数。
        std::size_t dofs_{}; ///< 局部维数。
        CudaBuffer<double> values_; ///< 设备局部贡献。
    };
    /** @brief 一批同序对流矩阵与载荷的组合。
     * @code auto c=integrateConvectionBatch(batch,h,ambient,scales); @endcode */
    class GpuConvectionBatch {
    public:
        /** @brief 设备对流矩阵；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.matrix(); @endcode */
        [[nodiscard]] const GpuLocalMatrixBatch& matrix() const noexcept { return matrix_; }
        /** @brief 设备环境载荷；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.load(); @endcode */
        [[nodiscard]] const GpuLocalVectorBatch& load() const noexcept { return load_; }

    private:
        friend class detail::GpuBatchAccess;
        GpuConvectionBatch(GpuLocalMatrixBatch m, GpuLocalVectorBatch f) : matrix_(std::move(m)), load_(std::move(f)) {}
        GpuLocalMatrixBatch matrix_; ///< 设备H。
        GpuLocalVectorBatch load_; ///< 设备f∞。
    };
    /** @brief 批量GPU积分导热。@param batch 设备几何批。
     * @param conductivity 按e*q+r排列的有限系数，取值范围与单表接口一致。
     * @param sectionScales 每实体二维厚度或三维1。
     * @pre 系数来源按同表点顺序，数量为size*pointCount，当前设备必须为批次设备。
     * @return 全部有限的设备局部结果；错误参数invalid_argument、数值失败EvaluationError、CUDA失败CudaError。
     * @note 同步错误检查只回读每实体状态，不回读全部矩阵；无逐单元kernel启动。
     * @code auto r=integrateConductivityBatch(batch,samples,scales); @endcode */
    [[nodiscard]] GpuLocalMatrixBatch integrateConductivityBatch(const GpuElementBatch& batch,
                                                                 std::span<const double> conductivity,
                                                                 std::span<const double> sectionScales);
    /** @brief 批量GPU积分一致容量。@param batch 设备几何批。
     * @param volumetricCapacity 按e*q+r排列的有限系数，取值范围与单表接口一致。
     * @param sectionScales 每实体二维厚度或三维1。
     * @pre 系数来源按同表点顺序，数量为size*pointCount，当前设备必须为批次设备。
     * @return 全部有限的设备局部结果；错误参数invalid_argument、数值失败EvaluationError、CUDA失败CudaError。
     * @note 同步错误检查只回读每实体状态，不回读全部矩阵；无逐单元kernel启动。
     * @code auto r=integrateCapacityBatch(batch,samples,scales); @endcode */
    [[nodiscard]] GpuLocalMatrixBatch integrateCapacityBatch(const GpuElementBatch& batch,
                                                             std::span<const double> volumetricCapacity,
                                                             std::span<const double> sectionScales);
    /** @brief 批量GPU积分体源。@param batch 设备几何批。
     * @param source 按e*q+r排列的有限系数，取值范围与单表接口一致。
     * @param sectionScales 每实体二维厚度或三维1。
     * @pre 系数来源按同表点顺序，数量为size*pointCount，当前设备必须为批次设备。
     * @return 全部有限的设备局部结果；错误参数invalid_argument、数值失败EvaluationError、CUDA失败CudaError。
     * @note 同步错误检查只回读每实体状态，不回读全部矩阵；无逐单元kernel启动。
     * @code auto r=integrateVolumeSourceBatch(batch,samples,scales); @endcode */
    [[nodiscard]] GpuLocalVectorBatch integrateVolumeSourceBatch(const GpuElementBatch& batch,
                                                                 std::span<const double> source,
                                                                 std::span<const double> sectionScales);
    /** @brief 批量GPU积分向外通量负载荷。@param batch 设备几何批。
     * @param outwardFlux 按e*q+r排列的有限系数，取值范围与单表接口一致。
     * @param sectionScales 每实体二维厚度或三维1。
     * @pre 系数来源按同表点顺序，数量为size*pointCount，当前设备必须为批次设备。
     * @return 全部有限的设备局部结果；错误参数invalid_argument、数值失败EvaluationError、CUDA失败CudaError。
     * @note 同步错误检查只回读每实体状态，不回读全部矩阵；无逐单元kernel启动。
     * @code auto r=integrateOutwardFluxBatch(batch,samples,scales); @endcode */
    [[nodiscard]] GpuLocalVectorBatch integrateOutwardFluxBatch(const GpuElementBatch& batch,
                                                                std::span<const double> outwardFlux,
                                                                std::span<const double> sectionScales);
    /** @brief 批量GPU积分对流矩阵和载荷。@param batch 设备几何批。
     * @param coefficient 按e*q+r排列的有限系数，取值范围与单表接口一致。
     * @param ambientTemperature 按点非负K值。
     * @param sectionScales 每实体二维厚度或三维1。
     * @pre 系数来源按同表点顺序，数量为size*pointCount，当前设备必须为批次设备。
     * @return 全部有限的设备局部结果；错误参数invalid_argument、数值失败EvaluationError、CUDA失败CudaError。
     * @note 同步错误检查只回读每实体状态，不回读全部矩阵；无逐单元kernel启动。
     * @code auto r=integrateConvectionBatch(batch,samples,ambient,scales); @endcode */
    [[nodiscard]] GpuConvectionBatch integrateConvectionBatch(const GpuElementBatch& batch,
                                                              std::span<const double> coefficient,
                                                              std::span<const double> ambientTemperature,
                                                              std::span<const double> sectionScales);
    /** @brief 在GPU上对同源局部向量逐项相加。@param batch 几何来源。
     * @param left 左贡献。@param right 右贡献。@pre 同一批来源、设备和形状。
     * @return 新设备向量，不改输入；非有限和抛EvaluationError。
     * @code auto sum=addLocalVectorBatches(batch,a,b); @endcode */
    [[nodiscard]] GpuLocalVectorBatch addLocalVectorBatches(const GpuElementBatch& batch,
                                                            const GpuLocalVectorBatch& left,
                                                            const GpuLocalVectorBatch& right);
} // namespace thermofem::thermal
