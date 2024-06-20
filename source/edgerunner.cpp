#include <string>

#include "edgerunner/edgerunner.hpp"

#include <fmt/core.h>

namespace edge
{

Model::Model()
    : m_name {fmt::format("{}", "edgerunner")}
{
}

auto Model::name() const -> char const*
{
    return m_name.c_str();
}

}  // namespace edge
