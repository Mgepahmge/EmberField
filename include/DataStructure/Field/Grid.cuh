/**
 * @file Grid.cuh
 * @brief 提供文件无关的可视化几何、来源标识及类型化数值数组。
 */
#pragma once
#include <DataStructure/Mesh/Mesh.cuh>
#include <variant>

namespace emberfield::field {
    /** @brief 保持标量类型的主机数值数组，64位标签不经过浮点转换。
     * @code NumericArray a{"temperature",1,3,"K",std::vector<double>{300,310,320}}; @endcode */
    struct NumericArray {
        std::string name_; ///< 稳定数组名。
        std::uint32_t components_{1}; ///< 每元组分量数。
        std::uint64_t tuples_{}; ///< 元组数。
        std::string unit_{"1"}; ///< SI单位或无量纲1。
        std::variant<std::vector<std::uint8_t>, std::vector<std::uint32_t>, std::vector<std::int64_t>,
                     std::vector<std::uint64_t>, std::vector<double>>
            values_; ///< AoS扁平数组。
        /** @brief 检查形状、字符及浮点有限性，无参数。
         * @pre 无。@note 无返回值；非法抛invalid_argument，溢出抛length_error。
         * @code a.validate(); @endcode */
        void validate() const;
        /** @brief 查询受检字节数，无参数。@pre 形状合法。
         * @return 数值载荷字节数；溢出length_error。@code auto n=a.sizeBytes(); @endcode */
        std::uint64_t sizeBytes() const;
    };

    /** @brief 校验XML可表达的UTF-8字符串。@param value 文本。@pre 无。
     * @note 无返回值；非法UTF-8或XML控制字符invalid_argument。@code validateText("steel"); @endcode */
    void validateText(std::string_view value);

    /** @brief 拥有活动输出点、单元布局和来源数组，不依赖VTK或物理求解器。
     * @details offsets含首项0；文件编码器负责转换为VTU末端偏移。
     * 标准容器RAII，复制深复制，移动后仅析构/赋值；只读对象可并发查询。
     * @code auto n=grid.pointCount(); @endcode */
    class VisualizationGrid {
    public:
        /** @brief 接管并校验完整可视化网格。
         * @param points 米制XYZ。@param offsets CSR行界限。@param connectivity 零基输出点下标。
         * @param layouts 每单元中立布局。@param pointIdentity 节点来源数组。
         * @param cellIdentity 单元来源数组。@param metadata 静态元数据。
         * @pre 非空网格，索引有效且数组形状一致。
         * @throws std::invalid_argument 非法形状；length_error溢出；资源异常传播。
         * @code auto g=VisualizationGrid(p,o,c,l,pi,ci,m); @endcode */
        VisualizationGrid(std::vector<std::array<double, 3>> points, std::vector<std::uint64_t> offsets,
                          std::vector<std::uint64_t> connectivity, std::vector<mesh::ElementType> layouts,
                          std::vector<NumericArray> pointIdentity, std::vector<NumericArray> cellIdentity,
                          std::vector<NumericArray> metadata);
        /** @brief 查询点数，无参数无前提。@return 点数。@code auto n=g.pointCount(); @endcode */
        std::size_t pointCount() const noexcept { return points_.size(); }
        /** @brief 查询单元数，无参数无前提。@return 单元数。@code auto n=g.cellCount(); @endcode */
        std::size_t cellCount() const noexcept { return layouts_.size(); }
        /** @brief 借用坐标，无参数无前提。@return 生命期内只读span。@code auto a=g.points(); @endcode */
        std::span<const std::array<double, 3>> points() const noexcept { return points_; }
        /** @brief 借用偏移，无参数无前提。@return 生命期内只读span。@code auto a=g.offsets(); @endcode */
        std::span<const std::uint64_t> offsets() const noexcept { return offsets_; }
        /** @brief 借用连接，无参数无前提。@return 生命期内只读span。@code auto a=g.connectivity(); @endcode */
        std::span<const std::uint64_t> connectivity() const noexcept { return connectivity_; }
        /** @brief 借用布局，无参数无前提。@return 生命期内只读span。@code auto a=g.layouts(); @endcode */
        std::span<const mesh::ElementType> layouts() const noexcept { return layouts_; }
        /** @brief 借用节点来源，无参数无前提。@return 只读span。@code auto a=g.pointIdentity(); @endcode */
        std::span<const NumericArray> pointIdentity() const noexcept { return pointIdentity_; }
        /** @brief 借用单元来源，无参数无前提。@return 只读span。@code auto a=g.cellIdentity(); @endcode */
        std::span<const NumericArray> cellIdentity() const noexcept { return cellIdentity_; }
        /** @brief 借用元数据，无参数无前提。@return 只读span。@code auto a=g.metadata(); @endcode */
        std::span<const NumericArray> metadata() const noexcept { return metadata_; }

    private:
        std::vector<std::array<double, 3>> points_; ///< 米制输出点。
        std::vector<std::uint64_t> offsets_; ///< 长C+1的行边界。
        std::vector<std::uint64_t> connectivity_; ///< 输出点索引。
        std::vector<mesh::ElementType> layouts_; ///< 文件无关的单元布局。
        std::vector<NumericArray> pointIdentity_; ///< 节点静态标识。
        std::vector<NumericArray> cellIdentity_; ///< 单元静态标识。
        std::vector<NumericArray> metadata_; ///< 静态属性及分组CSR。
    };
} // namespace emberfield::field
