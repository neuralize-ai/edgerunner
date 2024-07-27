#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

#include "edgerunner/edgerunner.hpp"

#include <nonstd/span.hpp>

#include "edgerunner/model.hpp"

#ifdef EDGERUNNER_TFLITE
#    include "edgerunner/tflite/model.hpp"
#endif

#ifdef EDGERUNNER_QNN
#    include "edgerunner/qnn/model.hpp"
#endif

namespace edge {

auto createModel(const std::filesystem::path& modelPath)
    -> std::unique_ptr<Model> {
    const auto modelExtension = modelPath.extension().string().substr(1);

    std::unique_ptr<Model> model;

#ifdef EDGERUNNER_TFLITE
    if (modelExtension == "tflite") {
        model = std::make_unique<tflite::ModelImpl>(modelPath);
    }
#endif

#ifdef EDGERUNNER_QNN
    if (modelExtension == "so" || modelExtension == "bin") {
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

#ifdef EDGERUNNER_TFLITE
    if (modelExtension == "tflite") {
        model = std::make_unique<tflite::ModelImpl>(modelBuffer);
    }
#endif

#ifdef EDGERUNNER_QNN
    if (modelExtension == "so" || modelExtension == "bin") {
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
