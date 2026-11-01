/** @file IntegrationFixture.cuh
 * @brief 创建有实际几何的单纯形空间供几何、热积分及GPU对照测试复用。
 */
#pragma once
#include <FiniteElement/Integration.cuh>

namespace integrationfixture {
    /** @brief 创建单个标准单纯形数据。@param d 维数2或3。@pre 合法维数。
     * @return 拥有数据。@code auto m=data(2); @endcode */
    inline thermofem::mesh::MeshData data(unsigned d = 2) {
        using namespace thermofem::mesh;
        MeshData m;
        m.nodes_ = {{91, {0, 0, 0}, {}, {}}, {37, {1, 0, 0}, {}, {}}, {10000000001ULL, {0, 1, 0}, {}, {}}};
        if (d == 3) {
            m.nodes_.push_back({15, {0, 0, 1}, {}, {}});
        }
        ElementBlock block;
        block.type_ = {d == 2 ? CellFamily::Triangle : CellFamily::Tetrahedron, static_cast<int>(d), d + 1, 1,
                       "complete"};
        block.tags_ = {200};
        block.offsets_ = {0, d + 1};
        for (std::size_t i = 0; i < d + 1; ++i) {
            block.nodes_.push_back(i);
        }
        m.elements_.push_back(block);
        return m;
    }
    /** @brief 由所有体单元建立空间。@param m 网格数据。@pre 仅支持单纯形体块。
     * @return 不可变空间；无效夹具抛logic_error。@code auto s=space(data()); @endcode */
    inline std::shared_ptr<const thermofem::fem::FiniteElementSpace> space(thermofem::mesh::MeshData m = data()) {
        using namespace thermofem;
        std::vector<mesh::CellRef> cells;
        std::vector<fem::ElementKey> keys;
        for (std::size_t b = 0; b < m.elements_.size(); ++b) {
            for (std::size_t e = 0; e < m.elements_[b].tags_.size(); ++e) {
                cells.push_back({b, e});
                keys.push_back({m.elements_[b].type_.family_});
            }
        }
        auto r = fem::buildFiniteElementSpace(std::make_shared<const mesh::Mesh>(std::move(m)), cells, keys);
        if (!r.space_) {
            throw std::logic_error("无效空间测试夹具");
        }
        return r.space_;
    }
    /** @brief 创建完整几何。@param m 数据。@param o 策略。@pre 维数与策略匹配。
     * @return 已准备几何。@code auto g=geometry(data()); @endcode */
    inline std::shared_ptr<const thermofem::fem::PreparedGeometry> geometry(thermofem::mesh::MeshData m = data(),
                                                                            thermofem::fem::GeometryOptions o = {}) {
        return thermofem::fem::buildPreparedGeometry(space(std::move(m)), o);
    }
} // namespace integrationfixture
