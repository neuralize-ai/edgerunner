#include "edgerunner/tflite/model.hpp"

#include <fmt/core.h>
#include <fmt/ranges.h>
#include <tensorflow/lite/core/c/c_api_types.h>
#include <tensorflow/lite/core/c/common.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>

namespace edge::tflite {

void ModelImpl::loadModel(const std::filesystem::path& modelPath) {
    m_modelPath = modelPath;
    setName(modelPath.stem());

    m_modelBuffer = ::tflite::FlatBufferModel::BuildFromFile(modelPath.c_str());

    ::tflite::ops::builtin::BuiltinOpResolver resolver;
    ::tflite::InterpreterBuilder(*m_modelBuffer, resolver)(&m_interpreter);

    m_interpreter->AllocateTensors();

    const auto metadata = m_modelBuffer->ReadAllMetadata();

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
