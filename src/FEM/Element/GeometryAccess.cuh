/** @file GeometryAccess.cuh @brief 内部几何工厂共享已核对来源的只读侧拓扑，不提供公开跳过校验标志。 */
#pragma once
#include <FEM/Element/Geometry.cuh>
/** @brief 受限几何构造与共享拓扑访问。 */
namespace emberfield::fem::detail {
    /** @brief 公开工厂与同源热准备链的受限构造访问。@code auto g=GeometryAccess::build(s,o,n,u,topology); @endcode */
    class GeometryAccess {
    public:
        /** @brief 准备数值几何，共享拓扑或自行建立。
         * @param space 空间。@param options 几何选项。@param maxCells 单元限额。
         * @param maxFacetUses 侧use限额。@param topology 内部已认证同网格同序活动单元的拓扑，空表示自行构建。
         * @pre 非空topology只能来自封闭绑定链，来源由调用层context验证，且无外部可变别名。
         * @return 完整只读几何；保留全部几何与邻接检查。
         * @code auto g=GeometryAccess::build(s,o,n,u,topology); @endcode */
        static std::shared_ptr<const PreparedGeometry> build(std::shared_ptr<const FiniteElementSpace> space,
                                                             const GeometryOptions& options, std::size_t maxCells,
                                                             std::size_t maxFacetUses,
                                                             std::shared_ptr<const mesh::FacetTopology> topology);
    };
} // namespace emberfield::fem::detail
