#pragma once

#include <tensorflow/lite/core/c/c_api_types.h>
#include <tensorflow/lite/core/c/common.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>

#include "edgerunner/model.hpp"

namespace edge::tflite {

class EDGERUNNER_EXPORT ModelImpl final : public Model {
  public:
    explicit ModelImpl(const std::filesystem::path& modelPath) {
        loadModel(modelPath);
    }

    ModelImpl(const ModelImpl&) = delete;
    ModelImpl(ModelImpl&&) = delete;
    auto operator=(const ModelImpl&) -> ModelImpl& = delete;
    auto operator=(ModelImpl&&) -> ModelImpl& = delete;

    ModelImpl() = default;

    ~ModelImpl() final;

    void loadModel(const std::filesystem::path& modelPath) final;

    auto applyDelegate(const DELEGATE& delegate) -> STATUS final;

    auto execute() -> STATUS final;

  private:
    void createInterpreter();

    void allocate();

    void deleteDelegate();

    EDGERUNNER_SUPPRESS_C4251
    std::filesystem::path m_modelPath;

    EDGERUNNER_SUPPRESS_C4251
    std::unique_ptr<::tflite::FlatBufferModel> m_modelBuffer;

    EDGERUNNER_SUPPRESS_C4251
    std::unique_ptr<::tflite::Interpreter> m_interpreter;

    EDGERUNNER_SUPPRESS_C4251
    TfLiteDelegate* m_delegate = nullptr;
};

}  // namespace edge::tflite
