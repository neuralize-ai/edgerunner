#include <fmt/core.h>
#include <fmt/ranges.h>
#include <tensorflow/lite/core/c/c_api_types.h>
#include <tensorflow/lite/core/c/common.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>

#include "edgerunner/model.hpp"
#include "tensor.hpp"

namespace edge::tflite {

class EDGERUNNER_EXPORT ModelImpl : public Model {
  public:
    explicit ModelImpl(const std::filesystem::path& modelPath) {
        loadModel(modelPath);
    }

    ModelImpl() = default;

    void loadModel(const std::filesystem::path& modelPath) final;

    void execute() final { m_interpreter->Invoke(); }

  private:
    EDGERUNNER_SUPPRESS_C4251
    std::filesystem::path m_modelPath;

    EDGERUNNER_SUPPRESS_C4251
    std::unique_ptr<::tflite::FlatBufferModel> m_modelBuffer;

    EDGERUNNER_SUPPRESS_C4251
    std::unique_ptr<::tflite::Interpreter> m_interpreter;
};

}  // namespace edge::tflite
