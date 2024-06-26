#pragma once

#include <filesystem>

#include "edgerunner/edgerunner_export.hpp"
#include "model.hpp"

namespace edge {

auto EDGERUNNER_EXPORT createModel(const std::filesystem::path& modelPath)
    -> std::unique_ptr<Model>;

}  // namespace edge
