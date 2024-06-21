#include <cstddef>
#include <filesystem>
#include <memory>

#include "edgerunner/tflite/model.hpp"

#include <tensorflow/lite/interpreter_builder.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model_builder.h>

namespace edge::tflite {

void ModelImpl::loadModel(const std::filesystem::path& modelPath) {
    m_modelPath = modelPath;
    setName(modelPath.stem());

    m_modelBuffer = ::tflite::FlatBufferModel::BuildFromFile(modelPath.c_str());

    const ::tflite::ops::builtin::BuiltinOpResolver resolver;
    ::tflite::InterpreterBuilder(*m_modelBuffer, resolver)(&m_interpreter);

    m_interpreter->AllocateTensors();

    const auto numInputs = m_interpreter->inputs().size();

    auto& inputs = accessInputs();
    inputs.reserve(numInputs);

    for (size_t i = 0; i < numInputs; ++i) {
        inputs.emplace_back(
            std::make_shared<TensorImpl>(m_interpreter->input_tensor(i)));
    }

    const auto numOutputs = m_interpreter->outputs().size();

    auto& outputs = accessOutputs();
    outputs.reserve(numOutputs);

    for (size_t i = 0; i < numOutputs; ++i) {
        outputs.emplace_back(
            std::make_shared<TensorImpl>(m_interpreter->output_tensor(i)));
    }
}

}  // namespace edge::tflite
