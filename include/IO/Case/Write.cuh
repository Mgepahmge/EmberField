/** @file Write.cuh @brief 已验证算例的规范JSON回显。 */
#pragma once
#include <Simulation/Case/Definition.cuh>
namespace emberfield::io {
    /** @brief 将完整有效配置序列化为JSON。@param definition 已通过validateCase的定义。
     * @pre 定义有效。@return UTF-8 JSON文本，不访问来源文件。资源异常传播。
     * @code auto json=serializeCase(c); @endcode */
    std::string serializeCase(const problem::CaseDefinition& definition);
} // namespace emberfield::io
