/** @file Vtu.cuh
 * @brief 声明不依赖CUDA执行或VTK库的主机VTU编码。
 */
#pragma once
#include <DataStructure/Field/Snapshot.cuh>
#include <ostream>

namespace emberfield::io {
    /** @brief VTU数值编码，原始追加区使用小端UInt64块头。 */
    enum class VtuEncoding { Ascii, AppendedRaw };
    /** @brief 写出完整VTU到借用流。
     * @param stream 可写流，二进制文件须binary模式。@param snapshot 完整网格与场。
     * @param encoding 编码方式。@pre 来源数组、温度和单元布局合法。
     * @note 无返回值，不关闭或原子发布流；写失败可能留下部分字节。
     * @throws std::invalid_argument 结构错误；domain_error不支持布局；length_error溢出；ios_base::failure写失败。
     * @code io::writeVtu(stream,frame,io::VtuEncoding::AppendedRaw); @endcode */
    void writeVtu(std::ostream& stream, const field::FieldSnapshot& snapshot,
                  VtuEncoding encoding = VtuEncoding::AppendedRaw);
} // namespace emberfield::io
