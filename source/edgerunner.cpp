#include "edgerunner/edgerunner.hpp"

#include "edgerunner/tflite/model.hpp"

namespace edge {

auto createModel(const std::filesystem::path& modelPath)
    -> std::unique_ptr<Model> {
    const auto modelExtension = modelPath.extension().string();

    if (modelExtension == "tflite") {
        return std::make_unique<tflite::ModelImpl>(modelPath);
    }

    /* unsupported */
    return nullptr;
}

}  // namespace edge
