/** @file Mesh.cuh
 * @brief 定义与文件编码和物理算例无关的宿主网格、分类实体和拓扑关联。
 */
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

/** @brief 提供独立于文件格式、离散算子及CUDA存储的网格模型。 */
namespace emberfield::mesh {
    using Tag = std::uint64_t; ///< 持久标签；不是连续数组索引。
    using Index = std::size_t; ///< 当前网格内的连续宿主索引。

    /** @brief 标识各维分类实体；同一标签在不同维度中可以重复。
     * @code EntityKey surface{2, 7}; @endcode */
    struct EntityKey {
        int dimension_{}; ///< 实体维度，0..3。
        std::int64_t tag_{}; ///< 实体标签，不将正负号混入实体身份。
        bool operator==(const EntityKey&) const = default; ///< 按维度和标签比较。
    };
    /** @brief 表示有向实体关联；允许仅载入局部网格时关联外部实体。
     * @code OrientedEntity edge{{1,3},-1}; @endcode */
    struct OrientedEntity {
        EntityKey entity_; ///< 被引用实体。
        int orientation_{1}; ///< +1或-1，不擅自翻转单元节点次序。
        bool operator==(const OrientedEntity&) const = default; ///< 值比较。
    };
    /** @brief 独立分类实体，可表达边界关联、物理分组及分区父实体。
     * @code Entity e; e.key_={2,1}; @endcode */
    struct Entity {
        EntityKey key_; ///< 实体身份。
        std::array<double, 6> bounds_{}; ///< 点使用前3值，其余实体为包围盒。
        std::vector<std::int64_t> physicalTags_; ///< 允许一个实体属于多个物理组。
        std::vector<OrientedEntity> boundary_; ///< 有序有向边界，不要求是封闭流形。
        std::vector<OrientedEntity> embedded_; ///< 嵌入实体，区别于拓扑边界。
        std::optional<EntityKey> parent_; ///< 分区实体的原始父分类，允许无父实体。
        std::vector<int> partitions_; ///< 所属分区集合。
        bool operator==(const Entity&) const = default; ///< 值比较。
    };
    /** @brief 网格节点，位置与分类参数坐标分离。
     * @code Node n; n.tag_=42; n.position_={0,0,0}; @endcode */
    struct Node {
        Tag tag_{}; ///< 正的持久标签，支持稀疏和大于32位的编号。
        std::array<double, 3> position_{}; ///< 三维嵌入坐标，低维网格也完整保存。
        std::optional<EntityKey> classification_; ///< 可缺省；不凭坐标猜测分类。
        std::vector<double> parameters_; ///< 分类实体的参数坐标，长度0或实体维度。
        bool operator==(const Node&) const = default; ///< 值比较。
    };
    /** @brief 表示参考单元的拓扑族，不限制为热传导首期支持的单元。 */
    enum class CellFamily {
        Point,
        Segment,
        Triangle,
        Quadrilateral,
        Tetrahedron,
        Hexahedron,
        Prism,
        Pyramid,
        Polygon,
        Polyhedron,
        Trihedron,
        Custom
    };
    /** @brief 中立的单元布局描述，文件类型编号由IO适配器映射。
     * @details 连通性次序属于referenceLayout的约定，不能按标签重新排序。
     * 自定义单元可注册自己的布局键，核心模型不依赖Gmsh编号。
     * @note 内置complete、edge-only等布局采用Gmsh参考单元局部节点次序；
     * 布局键是连接表语义，而非文件编号，不同节点次序必须使用不同布局键。
     * @code ElementType triangle{CellFamily::Triangle,2,3,1,"complete"}; @endcode */
    struct ElementType {
        CellFamily family_{CellFamily::Custom}; ///< 参考拓扑族。
        int dimension_{}; ///< 参考单元维度。
        std::size_t nodeCount_{}; ///< 固定节点数；0表示变长单元。
        int order_{}; ///< 多项式阶数；-1表示外部自定义或无法定义。
        std::string referenceLayout_; ///< 布局约定，如lagrange、edge-only、mini。
        bool operator==(const ElementType&) const = default; ///< 完整描述符比较。
    };
    /** @brief 同类单元的CSR式连接表，支持混合维度、任意阶和变长单元。
     * @code ElementBlock b; b.offsets_={0,3}; b.nodes_={0,1,2}; @endcode */
    struct ElementBlock {
        std::optional<EntityKey> classification_; ///< 所有单元的共同分类，可缺省。
        ElementType type_; ///< 参考拓扑和局部节点顺序约定。
        std::vector<Tag> tags_; ///< 单元持久标签。
        std::vector<Index> offsets_{0}; ///< 长度为单元数+1。
        std::vector<Index> nodes_; ///< 连续节点索引，绝不是文件节点标签。
        std::optional<std::vector<std::int64_t>> physicalTags_; ///< 可选块级分组覆盖；缺省继承实体分组。
        bool operator==(const ElementBlock&) const = default; ///< 值比较。
    };
    /** @brief 显式单元边界关联，可表达多面体面、非流形及不同维度单元关联。
     * @details 不要求所有边界都显式给出，不能用缺失记录推断物理外边界。
     * @code CellBoundary boundary{10,{{3,1},{4,-1}}}; @endcode */
    struct CellBoundary {
        Tag cell_{}; ///< 上维单元标签。
        std::vector<std::pair<Tag, int>> boundary_; ///< 下维单元标签及相对方向。
        bool operator==(const CellBoundary&) const = default; ///< 值比较。
    };
    /** @brief 物理组的可选名字；分组不包含材料或边界数值。
     * @code PhysicalGroup g{{2,5},"wall"}; @endcode */
    struct PhysicalGroup {
        EntityKey key_; ///< 维度和物理标签，不是几何实体标签。
        std::string name_; ///< 可含空格及UTF-8字符。
        bool operator==(const PhysicalGroup&) const = default; ///< 值比较。
    };
    /** @brief 周期拓扑关系，保存映射方向、仿射变换及对应节点。
     * @code PeriodicLink link; link.slave_={1,2}; link.master_={1,1}; @endcode */
    struct PeriodicLink {
        EntityKey slave_; ///< 从实体。
        EntityKey master_; ///< 主实体，与从实体同维。
        std::vector<double> affine_; ///< 空或行主序4x4齐次仿射矩阵。
        std::vector<std::pair<Index, Index>> nodes_; ///< 从节点、主节点的连续索引。
        bool operator==(const PeriodicLink&) const = default; ///< 值比较。
    };
    /** @brief 描述分布式网格的幽灵单元归属。
     * @code GhostElement ghost{12,1,{2}}; @endcode */
    struct GhostElement {
        Tag elementTag_{}; ///< 持久单元标签，可指当前局部文件外的单元。
        int owner_{}; ///< 拥有分区。
        std::vector<int> replicas_; ///< 具有幽灵副本的分区。
        bool operator==(const GhostElement&) const = default; ///< 值比较。
    };
    /** @brief 单元级分区成员关系；区别于分区实体分类和具有唯一拥有者的幽灵记录。
     * @code ElementPartition membership{12,{1},{2}}; @endcode */
    struct ElementPartition {
        Tag elementTag_{}; ///< 当前网格中的单元持久标签。
        std::vector<int> partitions_; ///< 正常所属分区。
        std::vector<int> ghostPartitions_; ///< 该单元作为幽灵副本的分区。
        bool operator==(const ElementPartition&) const = default; ///< 值比较。
    };
    /** @brief 离散实体的参数化采样，不是CAD模型或算例配置。
     * @details 曲线每样本为xyz,u；曲面为xyz,uv,最大及最小曲率方向，共11值。
     * 三角形连接使用本采样表的局部索引，不是网格节点标签。
     * @code Parametrization p; p.entity_={1,1}; @endcode */
    struct Parametrization {
        EntityKey entity_; ///< 维度1或2。
        std::vector<double> samples_; ///< 按样本连续排列的数值。
        std::vector<std::array<int, 3>> triangles_; ///< 曲面采样的三角划分，原始局部索引约定。
        bool operator==(const Parametrization&) const = default; ///< 值比较。
    };
    /** @brief 可编辑的构建值对象；不包含文件版本、编码或求解配置。
     * @code MeshData data; data.nodes_.push_back(node); Mesh mesh(std::move(data)); @endcode */
    struct MeshData {
        std::vector<Node> nodes_; ///< 节点顺序定义连续索引。
        std::vector<ElementBlock> elements_; ///< 混合单元块。
        std::vector<CellBoundary> cellBoundaries_; ///< 可选显式单元边界关联，区别于几何实体边界。
        std::vector<Entity> entities_; ///< 原始模型实体。
        std::vector<Entity> partitionedEntities_; ///< 分区后的分类实体。
        std::vector<PhysicalGroup> physicalGroups_; ///< 分组命名表。
        std::vector<PeriodicLink> periodic_; ///< 周期映射。
        std::size_t partitionCount_{}; ///< 全局分区数，0表示没有分区声明。
        std::vector<std::pair<std::int64_t, int>> ghostEntities_; ///< 幽灵实体标签及分区。
        std::vector<GhostElement> ghostElements_; ///< 幽灵单元元数据。
        std::vector<ElementPartition> elementPartitions_; ///< 独立于实体分类的单元级分区成员。
        std::vector<Parametrization> parametrizations_; ///< 离散参数化。
        bool operator==(const MeshData&) const = default; ///< 深值比较。
    };
    /** @brief 拥有经过验证的网格及标签索引；与文件IO完全分离。
     * @details 通过构建值对象整体创建，读取接口只读以保护连接表与标签索引。
     * 修改时复制data()、编辑后重建；不缓存不可靠的隐式相邻关系。
     * 允许悬挂节点、非流形网格、不连通区域、混合维度及不同高阶布局。
     * 本类不验证Jacobian正性、不建立自由度、不进行材料或边界数值解释。
     * @code Mesh m(std::move(data)); auto i=m.nodeIndex(42); @endcode */
    class Mesh {
    public:
        /** @brief 构造并验证完整网格。
         * @param data 所有权将被接管的构建数据。@pre 标签唯一、连接合法、数值有限。
         * @note 无返回值；失败抛invalid_argument或标准分配异常。
         * @code Mesh m(std::move(data)); @endcode */
        explicit Mesh(MeshData data = {});
        /** @brief 深复制网格和查询索引。@param other 来源。@pre 无。
         * @note 无返回值；分配失败抛标准异常。@code Mesh copy(original); @endcode */
        Mesh(const Mesh& other) = default;
        /** @brief 转移所有权，来源恢复为空网格。@param other 来源。@pre 无。
         * @note 无返回值，不抛异常。@code Mesh target(std::move(source)); @endcode */
        Mesh(Mesh&& other) noexcept;
        /** @brief 以复制交换提供强异常保证。@param other 来源。@pre 无。
         * @return 当前对象；失败保留原值。@code target=source; @endcode */
        Mesh& operator=(const Mesh& other);
        /** @brief 替换网格所有权，来源恢复为空；支持自移动。@param other 来源。@pre 无。
         * @return 当前对象，不抛异常。@code target=std::move(source); @endcode */
        Mesh& operator=(Mesh&& other) noexcept;
        /** @brief 读取不可变网格，无参数无额外前提。
         * @return 生命周期受网格限制的常量引用。@code auto& d=m.data(); @endcode */
        [[nodiscard]] const MeshData& data() const noexcept { return data_; }
        /** @brief 将持久标签解析为连续索引。
         * @param tag 正节点标签。@pre 标签存在。@return 连续索引，不存在抛out_of_range。
         * @code auto i=m.nodeIndex(42); @endcode */
        [[nodiscard]] Index nodeIndex(Tag tag) const;
        /** @brief 获取单元的连接视图。
         * @param block 块索引。@param element 块内单元索引。@pre 两索引有效。
         * @return 节点连续索引的只读span；越界抛out_of_range。@code auto c=m.connectivity(0,0); @endcode */
        [[nodiscard]] std::span<const Index> connectivity(Index block, Index element) const;

    private:
        /** @brief 交换数据与索引，维护两者一致性。@param other 另一网格。@pre 无。
         * @note 无返回值，不抛异常。@code swap(other); @endcode */
        void swap(Mesh& other) noexcept;
        MeshData data_; ///< 唯一拥有的网格值。
        std::unordered_map<Tag, Index> nodeIndices_; ///< 标签到连续节点索引的查询缓存。
    };
} // namespace emberfield::mesh
