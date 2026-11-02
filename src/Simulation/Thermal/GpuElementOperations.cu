/** @file GpuElementOperations.cu
 * @brief 实现复用设备积分表的大批量热算子及逐实体错误回传。
 */
#include <Simulation/Thermal/GpuElementOperations.cuh>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <limits>

namespace emberfield::thermal {
    namespace detail {
        /** @brief 保存批次来源身份与错误定位，禁止把同尺寸异源结果混加。 */
        struct GpuBatchIdentity {
            std::shared_ptr<const fem::FiniteElementSpace> space_; ///< 原空间生命周期。
            std::vector<mesh::CellRef> cells_; ///< 每行母元。
            std::vector<std::optional<std::size_t>> sides_; ///< 侧批局部侧，体批为空。
        };
        /** @brief 检查CUDA操作。@param status 返回码。@param operation 操作名。
         * @pre 无。@note 无返回值，失败抛CudaError。@code cudaCheck(cudaGetLastError(),"launch"); @endcode */
        void cudaCheck(cudaError_t status, const char* operation) {
            if (status != cudaSuccess) {
                throw CudaError(status, operation);
            }
        }
        /** @brief 数组大小受检乘法。@param a 大小。@param b 大小。@param limit 数组预算。
         * @pre 无。@return 乘积，否则length_error。@code auto n=product(e,q,limit); @endcode */
        std::size_t product(std::size_t a, std::size_t b, std::size_t limit) {
            if (b && a > limit / b) {
                throw std::length_error("GPU批数组超过限额或索引范围");
            }
            return a * b;
        }
        /** @brief GPU积分上传仅接受已校验的体积分表或侧积分表。 */
        template <class Table>
        concept GpuIntegrationTable =
            std::same_as<Table, fem::CellIntegrationData> || std::same_as<Table, fem::FacetIntegrationData>;
        /** @brief 内部接管与调度器，公共入口不暴露设备几何的可写指针。
         * @code auto b=GpuBatchAccess::upload(tables,limits); @endcode */
        class GpuBatchAccess {
        public:
            /** @brief 上传同形积分数据。@tparam Table 仅允许体或侧表。
             * @param tables 非空同源数组。@param limits 正限额。@pre 无。
             * @return 不可变设备批；非法形状/限额或资源错误传播。
             * @code auto b=upload(tables,limits); @endcode */
            template <GpuIntegrationTable Table>
            static GpuElementBatch upload(std::span<const Table> tables, GpuBatchLimits limits) {
                if (!limits.maxElements_ || !limits.maxScalars_ || tables.empty()) {
                    throw std::invalid_argument("GPU批必须非空且预算为正");
                }
                if (tables.size() > limits.maxElements_) {
                    throw std::length_error("GPU批实体数超过限额");
                }
                constexpr bool cell = std::same_as<Table, fem::CellIntegrationData>;
                GpuElementBatch batch;
                const auto& first = tables[0];
                batch.count_ = tables.size();
                batch.dofs_ = first.dofCount();
                batch.points_ = first.pointCount();
                batch.dimension_ = first.dimension();
                batch.exactDegree_ = first.exactDegree();
                batch.kind_ = cell ? GpuBatchKind::Cell : GpuBatchKind::Facet;
                if (batch.dofs_ > (cell ? 4 : 3) || !batch.dofs_ || batch.points_ > 4 || !batch.points_) {
                    throw std::invalid_argument("GPU批仅支持首期线性单纯形规则");
                }
                const auto nq = product(batch.count_, batch.points_, limits.maxScalars_);
                const auto nqn = product(nq, batch.dofs_, limits.maxScalars_);
                const auto nqnd = cell ? product(nqn, batch.dimension_, limits.maxScalars_) : 0;
                (void)product(batch.count_, batch.dofs_ * batch.dofs_, limits.maxScalars_);
                auto identity = std::make_shared<GpuBatchIdentity>();
                identity->space_ = first.space();
                identity->cells_.reserve(batch.count_);
                identity->sides_.reserve(batch.count_);
                std::vector<double> weights, values, gradients;
                weights.reserve(nq);
                values.reserve(nqn);
                gradients.reserve(nqnd);
                for (const auto& t : tables) {
                    if (t.space().get() != first.space().get() || t.dofCount() != batch.dofs_ ||
                        t.pointCount() != batch.points_ || t.dimension() != batch.dimension_ ||
                        t.exactDegree() != batch.exactDegree_) {
                        throw std::invalid_argument("GPU批的空间身份、维数、局部维数或规则不同");
                    }
                    weights.insert(weights.end(), t.physicalWeights().begin(), t.physicalWeights().end());
                    identity->cells_.push_back(t.space()->cells()[t.cellRow()]);
                    if constexpr (cell) {
                        values.insert(values.end(), t.values().begin(), t.values().end());
                        gradients.insert(gradients.end(), t.gradients().begin(), t.gradients().end());
                        identity->sides_.push_back({});
                    }
                    else {
                        values.insert(values.end(), t.traceValues().begin(), t.traceValues().end());
                        identity->sides_.push_back(t.use().localSide_);
                    }
                }
                batch.weights_ = CudaBuffer<double>(std::span<const double>(weights));
                batch.values_ = CudaBuffer<double>(std::span<const double>(values));
                batch.gradients_ = CudaBuffer<double>(std::span<const double>(gradients));
                batch.identity_ = std::move(identity);
                return batch;
            }
            /** @brief 校验设备和运行样本。@param b 几何批。@param values 系数。
             * @param scales 每实体厚度。@param kind 所需类型。@param degree 最低精度。@param range 范围。
             * @pre 无。@note 无返回值，非法参数invalid_argument，设备错误CudaError。
             * @code validate(b,k,s,GpuBatchKind::Cell,1,2); @endcode */
            static void validate(const GpuElementBatch& b, std::span<const double> values,
                                 std::span<const double> scales, GpuBatchKind kind, unsigned degree, int range) {
                if (!b.identity_ || !b.count_ || b.kind_ != kind || b.exactDegree_ < degree ||
                    values.size() != b.count_ * b.points_ || scales.size() != b.count_) {
                    throw std::invalid_argument("GPU批类型、精度或样本尺寸不相容");
                }
                int device = -1;
                cudaCheck(cudaGetDevice(&device), "cudaGetDevice");
                if (device != b.deviceId()) {
                    throw std::invalid_argument("GPU批属于不同CUDA设备");
                }
                for (auto v : values) {
                    if (!std::isfinite(v) || (range == 1 && v < 0) || (range == 2 && v <= 0)) {
                        throw std::invalid_argument("GPU系数样本不合法");
                    }
                }
                for (auto s : scales) {
                    if (!std::isfinite(s) || s <= 0 || (b.dimension_ == 3 && s != 1)) {
                        throw std::invalid_argument("GPU厚度/三维单位因子不合法");
                    }
                }
            }
            /** @brief 创建空输出设备方阵批。@param b 来源。@pre 有效批。
             * @return 候选输出。@code auto r=matrix(b); @endcode */
            static GpuLocalMatrixBatch matrix(const GpuElementBatch& b) {
                GpuLocalMatrixBatch r;
                r.identity_ = b.identity_;
                r.count_ = b.count_;
                r.dofs_ = b.dofs_;
                r.values_ = CudaBuffer<double>(emberfield::detail::checkedProduct(b.count_, b.dofs_ * b.dofs_));
                return r;
            }
            /** @brief 创建空输出设备向量批。@param b 来源。@pre 有效批。
             * @return 候选输出。@code auto r=vector(b); @endcode */
            static GpuLocalVectorBatch vector(const GpuElementBatch& b) {
                GpuLocalVectorBatch r;
                r.identity_ = b.identity_;
                r.count_ = b.count_;
                r.dofs_ = b.dofs_;
                r.values_ = CudaBuffer<double>(emberfield::detail::checkedProduct(b.count_, b.dofs_));
                return r;
            }
            /** @brief 同源性检查。@param b 来源。@param a 向量。@pre 无。
             * @note 无返回值，错配invalid_argument。@code same(b,a); @endcode */
            static void same(const GpuElementBatch& b, const GpuLocalVectorBatch& a) {
                if (b.identity_.get() != a.identity_.get() || b.count_ != a.count_ || b.dofs_ != a.dofs_ ||
                    b.deviceId() != a.values_.deviceId()) {
                    throw std::invalid_argument("GPU向量批来源身份不同");
                }
                int device = -1;
                cudaCheck(cudaGetDevice(&device), "cudaGetDevice");
                if (device != b.deviceId()) {
                    throw std::invalid_argument("当前设备与GPU批不同");
                }
            }
            /** @brief GPU计算后仅回读状态。@param b 来源。@param errors 每实体状态。
             * @pre 核函数已启动。@note 同步拷贝；失败抛带母元/侧的EvaluationError。
             * @code verify(b,errors); @endcode */
            static void verify(const GpuElementBatch& b, const CudaBuffer<int>& errors) {
                cudaCheck(cudaGetLastError(), "thermal batch kernel");
                std::vector<int> host(b.count_);
                errors.copyToHost(host);
                for (std::size_t e = 0; e < host.size(); ++e) {
                    if (host[e]) {
                        throw fem::EvaluationError(host[e] == 2 ? fem::EvaluationCode::NumericalUnderflow
                                                                : fem::EvaluationCode::NumericalOverflow,
                                                   fem::EvaluationStage::Element,
                                                   "GPU批量热积分不可表示；批行 " + std::to_string(e),
                                                   b.identity_->cells_[e], b.identity_->sides_[e]);
                    }
                }
            }
            /** @brief 启动一个独立积分算子。@param b 来源。@param op 算子0K/1M/2Q/3g/4对流。
             * @param coefficient 样本。@param ambient 对流温度或空。@param scales 厚度。
             * @param matrix 矩阵输出或空。@param vector 向量输出或空。@pre 调用入口已验证样本。
             * @note 无返回值，错误候选不向外发布。@code run(b,0,k,{},s,&m,nullptr); @endcode */
            static void run(const GpuElementBatch& b, int op, std::span<const double> coefficient,
                            std::span<const double> ambient, std::span<const double> scales,
                            GpuLocalMatrixBatch* matrix, GpuLocalVectorBatch* vector);
            /** @brief 组合对流。@param m 矩阵。@param f 载荷。@pre 同一来源。
             * @return 组合结果。@code auto c=convection(m,f); @endcode */
            static GpuConvectionBatch convection(GpuLocalMatrixBatch m, GpuLocalVectorBatch f) {
                return GpuConvectionBatch(std::move(m), std::move(f));
            }
            /** @brief 设备端相加。@param b 来源。@param a 左向量。@param c 右向量。
             * @pre 无。@return 新结果，失败不修改输入。@code auto r=add(b,a,c); @endcode */
            static GpuLocalVectorBatch add(const GpuElementBatch& b, const GpuLocalVectorBatch& a,
                                           const GpuLocalVectorBatch& c);
        };
    } // namespace detail
    namespace {
        /** @brief 以指数分解计算五因子乘积，避免非必要中间溢出。
         * @param a 因子。@param b 因子。@param c 因子。@param d 因子。@param e 因子。
         * @pre 输入有限。@return 最终double乘积，可由调用者检测范围。
         * @code auto v=scaledProduct(s,w,k,ga,gb); @endcode */
        __device__ double scaledProduct(double a, double b, double c, double d = 1, double e = 1) {
            const double factors[5] = {a, b, c, d, e};
            double mantissa = 1;
            int exponent = 0;
            for (int i = 0; i < 5; ++i) {
                if (factors[i] == 0) {
                    return 0;
                }
                int shift = 0;
                mantissa *= frexp(factors[i], &shift);
                exponent += shift;
            }
            return scalbn(mantissa, exponent);
        }
        /** @brief 检查设备浮点有限性。@param x 量。@param error 本实体状态。
         * @pre error初始化0。@note 无返回值，失败记录1。@code finite(x,error); @endcode */
        __device__ void finite(double x, int& error) {
            if (!isfinite(x)) {
                error = 1;
            }
        }
        /** @brief 按实体并行计算完整局部贡献，一个批次一个kernel，不逐单元调度。
         * @param count 实体数。@param n 局部DOF。@param qCount 点数。@param d 维数。
         * @param op 运算种类。@param weights 权重。@param values 基值。@param gradients 实际梯度。
         * @param coefficient 系数。@param ambient 环境温度。@param scales 厚度。
         * @param matrices 矩阵输出。@param loads 向量输出。@param errors 每实体状态。
         * @pre 已校验输入。@note 写完整输出，不返回值。
         * @code elementKernel<<<grid,128>>>(...); @endcode */
        __global__ void elementKernel(std::size_t count, unsigned n, unsigned qCount, unsigned d, int op,
                                      const double* weights, const double* values, const double* gradients,
                                      const double* coefficient, const double* ambient, const double* scales,
                                      double* matrices, double* loads, int* errors) {
            for (std::size_t e = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; e < count;
                 e += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
                double matrix[16] = {}, load[4] = {};
                int error = 0;
                bool positive = false, nonzero = false;
                for (unsigned q = 0; q < qCount; ++q) {
                    const auto point = e * qCount + q;
                    const auto base = point * n;
                    const double w = weights[point], c = coefficient[point], s = scales[e];
                    if (op == 4) {
                        positive |= c > 0;
                    }
                    for (unsigned a = 0; a < n; ++a) {
                        if (op == 2 || op == 3 || op == 4) {
                            const double term = scaledProduct(s, w, c, values[base + a],
                                                              op == 4       ? ambient[point]
                                                                  : op == 3 ? -1
                                                                            : 1);
                            finite(term, error);
                            load[a] += term;
                            finite(load[a], error);
                        }
                        if (op == 0 || op == 1 || op == 4) {
                            for (unsigned b = a; b < n; ++b) {
                                double term = 0;
                                if (op == 0) {
                                    for (unsigned k = 0; k < d; ++k) {
                                        const double component = scaledProduct(s, w, c, gradients[(base + a) * d + k],
                                                                               gradients[(base + b) * d + k]);
                                        finite(component, error);
                                        term += component;
                                        finite(term, error);
                                    }
                                }
                                else {
                                    term = scaledProduct(s, w, c, values[base + a], values[base + b]);
                                }
                                finite(term, error);
                                matrix[a * n + b] += term;
                                finite(matrix[a * n + b], error);
                                nonzero |= term != 0;
                            }
                        }
                    }
                }
                if (!error && (op == 0 || op == 1 || (op == 4 && positive)) && !nonzero) {
                    error = 2;
                }
                for (unsigned a = 0; a < n; ++a) {
                    if (loads) {
                        loads[e * n + a] = load[a];
                    }
                    if (matrices) {
                        for (unsigned b = a; b < n; ++b) {
                            matrices[e * n * n + a * n + b] = matrix[a * n + b];
                            matrices[e * n * n + b * n + a] = matrix[a * n + b];
                        }
                    }
                }
                errors[e] = error;
            }
        }
        /** @brief 同源向量逐实体相加并回报范围错误。
         * @param count 实体数。@param n 局部维数。@param a 左输入。@param b 右输入。
         * @param out 输出。@param errors 状态。@pre 形状及设备同源。
         * @note 写完整结果，无返回值。@code addKernel<<<grid,128>>>(...); @endcode */
        __global__ void addKernel(std::size_t count, unsigned n, const double* a, const double* b, double* out,
                                  int* errors) {
            for (std::size_t e = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; e < count;
                 e += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
                int error = 0;
                for (unsigned k = 0; k < n; ++k) {
                    const auto i = e * n + k;
                    out[i] = a[i] + b[i];
                    finite(out[i], error);
                }
                errors[e] = error;
            }
        }
        /** @brief 获取有界grid，内核使用size_t网格跨步。@param count 非零实体数。
         * @pre count>0。@return 至少一个block。@code auto grid=gridSize(count); @endcode */
        unsigned gridSize(std::size_t count) {
            return static_cast<unsigned>(std::min<std::size_t>((count - 1) / 128 + 1, 65535));
        }
    } // namespace
    void detail::GpuBatchAccess::run(const GpuElementBatch& b, int op, std::span<const double> coefficient,
                                     std::span<const double> ambient, std::span<const double> scales,
                                     GpuLocalMatrixBatch* matrix, GpuLocalVectorBatch* vector) {
        CudaBuffer<double> c(coefficient), s(scales), t(ambient);
        CudaBuffer<int> errors(b.count_);
        elementKernel<<<gridSize(b.count_), 128>>>(
            b.count_, static_cast<unsigned>(b.dofs_), static_cast<unsigned>(b.points_), b.dimension_, op,
            b.weights_.deviceData(), b.values_.deviceData(), b.gradients_.deviceData(), c.deviceData(), t.deviceData(),
            s.deviceData(), matrix ? matrix->values_.deviceData() : nullptr,
            vector ? vector->values_.deviceData() : nullptr, errors.deviceData());
        verify(b, errors);
    }
    GpuLocalVectorBatch detail::GpuBatchAccess::add(const GpuElementBatch& b, const GpuLocalVectorBatch& a,
                                                    const GpuLocalVectorBatch& c) {
        same(b, a);
        same(b, c);
        auto result = vector(b);
        CudaBuffer<int> errors(b.count_);
        addKernel<<<gridSize(b.count_), 128>>>(b.count_, static_cast<unsigned>(b.dofs_), a.values_.deviceData(),
                                               c.values_.deviceData(), result.values_.deviceData(),
                                               errors.deviceData());
        verify(b, errors);
        return result;
    }
    GpuElementBatch uploadCellBatch(std::span<const fem::CellIntegrationData> t, GpuBatchLimits limits) {
        return detail::GpuBatchAccess::upload(t, limits);
    }
    GpuElementBatch uploadFacetBatch(std::span<const fem::FacetIntegrationData> t, GpuBatchLimits limits) {
        return detail::GpuBatchAccess::upload(t, limits);
    }
    std::vector<double> GpuLocalMatrixBatch::copyToHost() const {
        std::vector<double> host(values_.size());
        values_.copyToHost(host);
        return host;
    }
    std::vector<double> GpuLocalVectorBatch::copyToHost() const {
        std::vector<double> host(values_.size());
        values_.copyToHost(host);
        return host;
    }
    GpuLocalMatrixBatch integrateConductivityBatch(const GpuElementBatch& b, std::span<const double> k,
                                                   std::span<const double> s) {
        detail::GpuBatchAccess::validate(b, k, s, GpuBatchKind::Cell, 1, 2);
        auto r = detail::GpuBatchAccess::matrix(b);
        detail::GpuBatchAccess::run(b, 0, k, {}, s, &r, nullptr);
        return r;
    }
    GpuLocalMatrixBatch integrateCapacityBatch(const GpuElementBatch& b, std::span<const double> c,
                                               std::span<const double> s) {
        detail::GpuBatchAccess::validate(b, c, s, GpuBatchKind::Cell, 2, 2);
        auto r = detail::GpuBatchAccess::matrix(b);
        detail::GpuBatchAccess::run(b, 1, c, {}, s, &r, nullptr);
        return r;
    }
    GpuLocalVectorBatch integrateVolumeSourceBatch(const GpuElementBatch& b, std::span<const double> q,
                                                   std::span<const double> s) {
        detail::GpuBatchAccess::validate(b, q, s, GpuBatchKind::Cell, 1, 0);
        auto r = detail::GpuBatchAccess::vector(b);
        detail::GpuBatchAccess::run(b, 2, q, {}, s, nullptr, &r);
        return r;
    }
    GpuLocalVectorBatch integrateOutwardFluxBatch(const GpuElementBatch& b, std::span<const double> g,
                                                  std::span<const double> s) {
        detail::GpuBatchAccess::validate(b, g, s, GpuBatchKind::Facet, 1, 0);
        auto r = detail::GpuBatchAccess::vector(b);
        detail::GpuBatchAccess::run(b, 3, g, {}, s, nullptr, &r);
        return r;
    }
    GpuConvectionBatch integrateConvectionBatch(const GpuElementBatch& b, std::span<const double> h,
                                                std::span<const double> t, std::span<const double> s) {
        detail::GpuBatchAccess::validate(b, h, s, GpuBatchKind::Facet, 2, 1);
        detail::GpuBatchAccess::validate(b, t, s, GpuBatchKind::Facet, 2, 1);
        auto m = detail::GpuBatchAccess::matrix(b);
        auto f = detail::GpuBatchAccess::vector(b);
        detail::GpuBatchAccess::run(b, 4, h, t, s, &m, &f);
        return detail::GpuBatchAccess::convection(std::move(m), std::move(f));
    }
    GpuLocalVectorBatch addLocalVectorBatches(const GpuElementBatch& b, const GpuLocalVectorBatch& a,
                                              const GpuLocalVectorBatch& c) {
        return detail::GpuBatchAccess::add(b, a, c);
    }
} // namespace emberfield::thermal
