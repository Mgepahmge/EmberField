/** @file CaseFixture.cuh
 * @brief 为算例语义测试构造两个三角形及显式边界，不依赖文件或GPU。
 */
#pragma once
#include <Simulation/Case/Binding.cuh>
#include <algorithm>
#include <memory>
namespace casefixture {
    /** @brief 创建标签选择器。@param dimension 维数。@param tag 标签。@pre 无。
     * @return 选择器。@code auto r=group(2,11); @endcode */
    inline emberfield::problem::RegionSelector group(int dimension, std::int64_t tag) {
        emberfield::problem::RegionSelector s;
        s.dimension_ = dimension;
        s.tag_ = tag;
        return s;
    }
    /** @brief 创建一份活动域中包含两三角形的网格。@pre 无。
     * @return 构建数据。@code auto data=meshData(); @endcode */
    inline emberfield::mesh::MeshData meshData() {
        using namespace emberfield::mesh;
        MeshData d;
        d.nodes_ = {{5000000001ULL, {0, 0, 0}, {}, {}},
                    {8, {1, 0, 0}, {}, {}},
                    {42, {1, 1, 0}, {}, {}},
                    {7, {0, 1, 0}, {}, {}}};
        ElementBlock body;
        body.type_ = {CellFamily::Triangle, 2, 3, 1, "complete"};
        body.tags_ = {100, 300};
        body.offsets_ = {0, 3, 6};
        body.nodes_ = {0, 1, 2, 0, 2, 3};
        body.physicalTags_ = std::vector<std::int64_t>{11, 9007199254740993LL};
        d.elements_.push_back(body);
        for (std::size_t i = 0; i < 5; ++i) {
            const std::vector<std::vector<Index>> connections{{0, 1}, {1, 2}, {2, 3}, {3, 0}, {0, 2}};
            ElementBlock edge;
            edge.type_ = {CellFamily::Segment, 1, 2, 1, "complete"};
            edge.tags_ = {400 + i};
            edge.offsets_ = {0, 2};
            edge.nodes_ = connections[i];
            edge.physicalTags_ = std::vector<std::int64_t>{static_cast<std::int64_t>(21 + i)};
            d.elements_.push_back(edge);
        }
        d.physicalGroups_ = {
            {{2, 11}, "body"}, {{1, 21}, "bottom"}, {{1, 22}, "right"}, {{1, 23}, "top"}, {{1, 24}, "left"}};
        return d;
    }
    /** @brief 创建合法稳态二维算例。@pre 无。
     * @return 独立定义。@code auto c=definition(); @endcode */
    inline emberfield::problem::CaseDefinition definition() {
        using namespace emberfield::problem;
        CaseDefinition c;
        c.analysis_.thickness_ = 0.01;
        c.analysis_.domain_ = group(2, 11);
        Material material;
        material.id_ = "solid";
        material.conductivity_ = ConstantScalar{15};
        c.materials_.push_back(material);
        c.materialAssignments_.push_back({"all", group(2, 11), "solid"});
        c.boundaries_.push_back({"left", group(1, 24), TemperatureBoundary{ConstantScalar{300}}});
        return c;
    }
    /** @brief 将定义转换为合法瞬态。@param c 待修改定义。@pre 有材料。
     * @note 无返回值。@code makeTransient(c); @endcode */
    inline void makeTransient(emberfield::problem::CaseDefinition& c) {
        using namespace emberfield::problem;
        c.analysis_.type_ = AnalysisType::Transient;
        c.initialState_ = InitialState{ConstantScalar{300}, {}};
        c.time_ = TimeOptions{0, 10, 0.1, 1000};
        for (auto& m : c.materials_) {
            m.volumetricHeatCapacity_ = ConstantScalar{3900000};
        }
    }
    /** @brief 检查诊断码。@param d 诊断。@param code 目标码。@pre 无。
     * @return 是否存在。@code auto found=hasCode(d,"MATERIAL_GAP"); @endcode */
    inline bool hasCode(const emberfield::problem::Diagnostics& d, const std::string& code) {
        return std::any_of(d.entries().begin(), d.entries().end(), [&](const auto& v) { return v.code_ == code; });
    }
    /** @brief 绑定测试定义。@param c 定义。@param d 网格数据。@pre 网格合法。
     * @return 绑定结果。@code auto r=bind(definition()); @endcode */
    inline emberfield::problem::BindResult bind(const emberfield::problem::CaseDefinition& c,
                                                emberfield::mesh::MeshData d = meshData()) {
        return emberfield::problem::bindCase(c, std::make_shared<const emberfield::mesh::Mesh>(std::move(d)));
    }
} // namespace casefixture
