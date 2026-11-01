/** @file msh_test.cpp
 * @brief 覆盖MSH文本、二进制、旧版、元数据和损坏输入，所有用例在宿主执行。
 */
#include <IO/Msh.cuh>
#include <algorithm>
#include <bit>
#include <chrono>
#include <cstring>
#include <fstream>
#include <gtest/gtest.h>
#include <sstream>
#include <stdexcept>
namespace {
    using namespace thermofem::mesh;
    using namespace thermofem::io;
    /** @brief 提供手写4.1输入。@pre 无。@return 文本。
     * @code auto input=fixture(); @endcode */
    std::string fixture() {
        return R"($MeshFormat
4.1 0 8
$EndMeshFormat
$PhysicalNames
1
2 8 "heated domain"
$EndPhysicalNames
$Entities
0 0 1 0
7 0 0 0 1 1 0 1 8 0
$EndEntities
$Nodes
1 3 2 5000000000
2 7 1 3
90 2 5000000000
0 0 0 0 0
1 0 0 1 0
0 1 0 0 1
$EndNodes
$Elements
1 1 31 31
2 7 2 1
31 90 2 5000000000
$EndElements
$Periodic
1
1 2 1
0
1
90 2
$EndPeriodic
)";
    }
    /** @brief 从字符串解析文档。@param text 输入。@pre 合法输入。@return 文档。
     * @code auto d=parse(fixture()); @endcode */
    MshDocument parse(const std::string& text) {
        std::istringstream in(text, std::ios::in | std::ios::binary);
        return readMsh(in);
    }
    /** @brief 往返测试。@param d 文档。@param e 编码。@param v 版本。@pre 可表示。
     * @return 独立文档。@code auto r=roundtrip(d,MshEncoding::Ascii); @endcode */
    MshDocument roundtrip(const MshDocument& d, MshEncoding e, MshVersion v = MshVersion::V41) {
        std::ostringstream out(std::ios::out | std::ios::binary);
        writeMsh(out, d, e, v);
        return parse(out.str());
    }
    TEST(Msh, ReadsIndependentAsciiAndPreservesNodeOrder) {
        auto d = parse(fixture());
        EXPECT_EQ(d.mesh_.nodeIndex(5000000000ULL), 2);
        EXPECT_EQ(d.mesh_.connectivity(0, 0)[1], 1);
        EXPECT_EQ(d.mesh_.data().nodes_[1].parameters_, (std::vector<double>{1, 0}));
        EXPECT_EQ(d.mesh_.data().physicalGroups_[0].name_, "heated domain");
    }
    TEST(Msh, RoundtripsRichMetadataInBothEncodings) {
        auto d = parse(fixture());
        auto data = d.mesh_.data();
        data.partitionCount_ = 2;
        Entity e;
        e.key_ = {2, 11};
        e.parent_ = EntityKey{2, 7};
        e.partitions_ = {1, 2};
        e.bounds_ = {0, 0, 0, 1, 1, 0};
        data.partitionedEntities_ = {e};
        data.ghostEntities_ = {{12, 2}};
        data.ghostElements_ = {{31, 1, {2}}};
        Parametrization p;
        p.entity_ = {2, 7};
        p.samples_ = std::vector<double>(33, 0);
        p.triangles_ = {{{0, 1, 2}}};
        data.parametrizations_.push_back(std::move(p));
        d.mesh_ = Mesh(std::move(data));
        for (auto encoding : {MshEncoding::Ascii, MshEncoding::Binary}) {
            EXPECT_EQ(roundtrip(d, encoding).mesh_.data(), d.mesh_.data());
        }
    }
    TEST(Msh, SupportsHighOrderAndAllRegisteredFixedLayouts) {
        for (int code = 1; code <= 140; ++code) {
            if (code == 34 || code == 35 || code == 69 || (code >= 76 && code <= 78)) {
                continue;
            }
            auto type = mshElementType(code);
            MeshData data;
            ElementBlock b;
            b.classification_ = EntityKey{type.dimension_, 1};
            b.type_ = type;
            b.tags_ = {1};
            for (Index i = 0; i < type.nodeCount_; ++i) {
                Node n;
                n.tag_ = i + 1;
                n.classification_ = b.classification_;
                data.nodes_.push_back(n);
                b.nodes_.push_back(i);
            }
            b.offsets_ = {0, b.nodes_.size()};
            data.elements_ = {b};
            MshDocument d;
            d.mesh_ = Mesh(std::move(data));
            EXPECT_EQ(roundtrip(d, MshEncoding::Binary).mesh_.data(), d.mesh_.data()) << code;
        }
    }
    TEST(Msh, SupportsRepeatedNodeSections) {
        auto text = fixture();
        auto begin = text.find("$Nodes"), end = text.find("$EndNodes") + 10;
        text.replace(begin, end - begin, R"($Nodes
1 1 90 90
2 7 1 1
90
0 0 0 0 0
$EndNodes
$Nodes
1 2 2 5000000000
2 7 1 2
2 5000000000
1 0 0 1 0
0 1 0 0 1
$EndNodes
)");
        EXPECT_EQ(parse(text).mesh_.data(), parse(fixture()).mesh_.data());
    }
    TEST(Msh, ReadsAndSavesLegacyTagsAndPeriodicLinks) {
        const auto d = parse(R"($MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
3
9 0 0 0
2 1 0 0
7 0 1 0
$EndNodes
$Elements
1
4 2 4 5 3 1 1 9 2 7
$EndElements
$Periodic
1
1 2 1
1
9 2
$EndPeriodic
)");
        EXPECT_EQ(d.legacyElementTags_.at(4), (std::vector<int>{5, 3, 1, 1}));
        for (auto encoding : {MshEncoding::Ascii, MshEncoding::Binary}) {
            auto result = roundtrip(d, encoding, MshVersion::V22);
            EXPECT_EQ(result.mesh_.data(), d.mesh_.data());
            EXPECT_EQ(result.legacyElementTags_, d.legacyElementTags_);
        }
    }
    TEST(Msh, EmptyMeshBothVersionsAndEncodings) {
        for (auto v : {MshVersion::V22, MshVersion::V41}) {
            for (auto e : {MshEncoding::Ascii, MshEncoding::Binary}) {
                MshDocument empty;
                EXPECT_TRUE(roundtrip(empty, e, v).mesh_.data().nodes_.empty());
            }
        }
    }
    /** @brief 手工构造指定端序、指定宽度的二进制网格。@tparam T 数值类型。
     * @param out 流。@param value 数值。@pre 算术类型。@note 无返回值。
     * @code appendBig(out,1); @endcode */
    template <class T>
        requires std::is_arithmetic_v<T>
    void appendBig(std::ostream& out, T value) {
        std::array<char, sizeof(T)> b;
        std::memcpy(b.data(), &value, sizeof(T));
        if (std::endian::native == std::endian::little) {
            std::reverse(b.begin(), b.end());
        }
        out.write(b.data(), b.size());
    }
    TEST(Msh, ReadsIndependentBigEndian32BitSizeFixture) {
        std::ostringstream out(std::ios::out | std::ios::binary);
        out << "$MeshFormat\n4.1 1 4\n";
        appendBig(out, 1);
        out << "\n$EndMeshFormat\n$Nodes\n";
        for (unsigned i : {1, 1, 42, 42}) {
            appendBig(out, i);
        }
        for (int i : {0, 1, 0}) {
            appendBig(out, i);
        }
        appendBig(out, 1U);
        appendBig(out, 42U);
        for (double x : {1., 2., 3.}) {
            appendBig(out, x);
        }
        out << "\n$EndNodes\n$Elements\n";
        for (unsigned i : {1, 1, 8, 8}) {
            appendBig(out, i);
        }
        for (int i : {0, 1, 15}) {
            appendBig(out, i);
        }
        for (unsigned i : {1, 8, 42}) {
            appendBig(out, i);
        }
        out << "\n$EndElements\n";
        auto d = parse(out.str());
        EXPECT_EQ(d.mesh_.data().nodes_[0].position_, (std::array<double, 3>{1, 2, 3}));
        EXPECT_EQ(d.mesh_.connectivity(0, 0)[0], 0);
    }
    TEST(Msh, RetainsAsciiExtensionsAndRejectsBinaryAmbiguity) {
        auto d = parse(fixture() + "$Comments\nhello\n$EndComments\n");
        EXPECT_EQ(roundtrip(d, MshEncoding::Ascii).extensions_[0].text_, "hello\n");
        EXPECT_THROW(roundtrip(d, MshEncoding::Binary), std::runtime_error);
    }
    TEST(Msh, RejectsCorruptionAndResourceOverruns) {
        auto text = fixture();
        text.replace(text.find("31 90 2"), 7, "31 91 2");
        EXPECT_THROW(parse(text), std::runtime_error);
        text = fixture();
        text.replace(text.find("1 3 2 5000000000"), 15, "1 4 2 5000000000");
        EXPECT_THROW(parse(text), std::runtime_error);
        EXPECT_THROW(parse(fixture().substr(0, 80)), std::runtime_error);
        EXPECT_THROW(parse(fixture() + "$NodeData\n"), std::runtime_error);
        std::istringstream in(fixture());
        EXPECT_THROW(readMsh(in, {2, 10000}), std::runtime_error);
        std::istringstream small(fixture());
        EXPECT_THROW(readMsh(small, {100, 32}), std::runtime_error);
    }
    TEST(Msh, RejectsUnrepresentableTopologyBeforeTouchingOutput) {
        auto d = parse(fixture());
        auto data = d.mesh_.data();
        data.elements_[0].type_.referenceLayout_ = "custom";
        d.mesh_ = Mesh(std::move(data));
        std::ostringstream out;
        out << "sentinel";
        EXPECT_THROW(writeMsh(out, d), std::runtime_error);
        EXPECT_EQ(out.str(), "sentinel");
    }
    TEST(Msh, PreservesExistingFileWhenPreflightFails) {
        const auto path = std::filesystem::temp_directory_path() /
            ("thermofem-preflight-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
             ".msh");
        {
            std::ofstream out(path);
            out << "sentinel";
        }
        auto d = parse(fixture());
        auto data = d.mesh_.data();
        data.nodes_[0].classification_.reset();
        data.nodes_[0].parameters_.clear();
        d.mesh_ = Mesh(std::move(data));
        EXPECT_THROW(saveMsh(path, d), std::runtime_error);
        std::ifstream input(path);
        std::string contents;
        input >> contents;
        EXPECT_EQ(contents, "sentinel");
        input.close();
        std::filesystem::remove(path);
    }
    TEST(Msh, AcceptsConsistentRepeatedEntitiesAndRejectsConflicts) {
        const std::string entities = "$Entities\n0 0 1 0\n7 0 0 0 1 1 0 1 8 0\n$EndEntities\n";
        EXPECT_EQ(parse(fixture() + entities).mesh_.data(), parse(fixture()).mesh_.data());
        auto conflicting = entities;
        conflicting.replace(conflicting.find("1 8 0"), 5, "1 9 0");
        EXPECT_THROW(parse(fixture() + conflicting), std::runtime_error);
    }
    TEST(Msh, AcceptsRepeatedElementSectionsButRejectsDuplicateCellIdentity) {
        const std::string section = "$Elements\n1 1 32 32\n2 7 2 1\n32 90 2 5000000000\n$EndElements\n";
        auto d = parse(fixture() + section);
        EXPECT_EQ(d.mesh_.data().elements_.size(), 2);
        EXPECT_EQ(d.mesh_.data().elements_[1].tags_[0], 32);
        auto repeated = fixture();
        repeated +=
            repeated.substr(repeated.find("$Elements"), repeated.find("$Periodic") - repeated.find("$Elements"));
        EXPECT_THROW(parse(repeated), std::runtime_error);
    }
    TEST(Msh, RejectsTruncatedBinaryPayloadAndInvalidEncoding) {
        auto d = parse(fixture());
        std::ostringstream out(std::ios::out | std::ios::binary);
        writeMsh(out, d, MshEncoding::Binary);
        const auto text = out.str();
        EXPECT_THROW(parse(text.substr(0, text.find("$Elements") + 25)), std::runtime_error);
        EXPECT_THROW(writeMsh(out, d, static_cast<MshEncoding>(42)), std::runtime_error);
        EXPECT_THROW(parse("$MeshFormat\n4.2 0 8\n$EndMeshFormat\n"), std::runtime_error);
    }
    TEST(Msh, RejectsLegacyPartitionCorruptionAndMetadataLoss) {
        const std::string text = "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n1\n1 0 0 0\n$EndNodes\n$Elements\n1\n1 "
                                 "15 4 5 1 1 -2 1\n$EndElements\n";
        auto d = parse(text);
        EXPECT_EQ(d.mesh_.data().elementPartitions_[0].ghostPartitions_, (std::vector<int>{2}));
        EXPECT_EQ(roundtrip(d, MshEncoding::Binary, MshVersion::V22).mesh_.data(), d.mesh_.data());
        auto data = d.mesh_.data();
        data.entities_[0].physicalTags_.push_back(6);
        d.mesh_ = Mesh(std::move(data));
        EXPECT_THROW(roundtrip(d, MshEncoding::Ascii, MshVersion::V22), std::runtime_error);
        auto corrupt = text;
        corrupt.replace(corrupt.find("5 1 1 -2"), 8, "5 1 3 -2");
        EXPECT_THROW(parse(corrupt), std::runtime_error);
    }
} // namespace
