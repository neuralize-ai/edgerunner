#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

#include "edgerunner/edgerunner.hpp"

#include <nonstd/span.hpp>

#include "edgerunner/model.hpp"
#include "edgerunner/qnn/model.hpp"
#include "edgerunner/tflite/model.hpp"

namespace edge {

auto createModel(const std::filesystem::path& modelPath)
    -> std::unique_ptr<Model> {
    const auto modelExtension = modelPath.extension().string().substr(1);

    if (modelExtension == "tflite") {
        return std::make_unique<tflite::ModelImpl>(modelPath);
    }
    if (modelExtension == "so") {
        return std::make_unique<qnn::ModelImpl>(modelPath);
    }

    /* unsupported */
    return nullptr;
}

auto createModel(const nonstd::span<uint8_t>& modelBuffer,
                 const std::string& modelExtension) -> std::unique_ptr<Model> {
    if (modelExtension == "tflite") {
        return std::make_unique<tflite::ModelImpl>(modelBuffer);
    }
    if (modelExtension == "so") {
        return std::make_unique<qnn::ModelImpl>(modelBuffer);
    }

    /* unsupported */
    return nullptr;
}

}  // namespace edge
