#include <filesystem>
#include <memory>

#include "edgerunner/edgerunner.hpp"

#include "edgerunner/model.hpp"
#include "edgerunner/tflite/model.hpp"

namespace edge {

auto createModel(const std::filesystem::path& modelPath)
    -> std::unique_ptr<Model> {
    const auto modelExtension = modelPath.extension().string().substr(1);

    if (modelExtension == "tflite") {
        return std::make_unique<tflite::ModelImpl>(modelPath);
    }

    /* unsupported */
    return nullptr;
}

auto createModel(const nonstd::span<uint8_t>& modelBuffer,
                 const std::string& modelExtension) -> std::unique_ptr<Model> {
    if (modelExtension == "tflite") {
        return std::make_unique<tflite::ModelImpl>(modelBuffer);
    }

    /* unsupported */
    return nullptr;
}

}  // namespace edge
