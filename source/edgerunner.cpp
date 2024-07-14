#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

#include "edgerunner/edgerunner.hpp"

#include <nonstd/span.hpp>

#include "edgerunner/model.hpp"
#include "edgerunner/tflite/model.hpp"

#ifdef EDGERUNNER_QNN
#    include "edgerunner/qnn/model.hpp"
#endif

namespace edge {

auto createModel(const std::filesystem::path& modelPath)
    -> std::unique_ptr<Model> {
    const auto modelExtension = modelPath.extension().string().substr(1);

    std::unique_ptr<Model> model;

    if (modelExtension == "tflite") {
        model = std::make_unique<tflite::ModelImpl>(modelPath);
    }

#ifdef EDGERUNNER_QNN
    if (modelExtension == "so") {
        model = std::make_unique<qnn::ModelImpl>(modelPath);
    }
#endif

    if (model != nullptr && model->getCreationStatus() == STATUS::SUCCESS) {
        return model;
    }

    /* unsupported or failed */
    return nullptr;
}

auto createModel(const nonstd::span<uint8_t>& modelBuffer,
                 const std::string& modelExtension) -> std::unique_ptr<Model> {
    std::unique_ptr<Model> model;

    if (modelExtension == "tflite") {
        model = std::make_unique<tflite::ModelImpl>(modelBuffer);
    }

#ifdef EDGERUNNER_QNN
    if (modelExtension == "so") {
        model = std::make_unique<qnn::ModelImpl>(modelBuffer);
    }
#endif

    if (model != nullptr && model->getCreationStatus() == STATUS::SUCCESS) {
        return model;
    }

    /* unsupported or failed */
    return nullptr;
}

}  // namespace edge
