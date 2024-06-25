#include <cstddef>
#include <filesystem>
#include <memory>

#include "edgerunner/tflite/model.hpp"

#include <tensorflow/lite/core/c/c_api_types.h>
#include <tensorflow/lite/interpreter_builder.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model_builder.h>

#include "edgerunner/tflite/tensor.hpp"

#ifdef EDGERUNNER_GPU
#    include <tensorflow/lite/delegates/gpu/delegate.h>
#endif

#include "edgerunner/model.hpp"

namespace edge::tflite {

void ModelImpl::loadModel(const std::filesystem::path& modelPath) {
    m_modelPath = modelPath;
    setName(modelPath.stem().string());

    m_modelBuffer = ::tflite::FlatBufferModel::BuildFromFile(modelPath.c_str());

    createInterpreter();

    allocate();
}

void ModelImpl::createInterpreter() {
    const ::tflite::ops::builtin::BuiltinOpResolver opResolver;
    ::tflite::InterpreterBuilder(*m_modelBuffer, opResolver)(&m_interpreter);
}

void ModelImpl::allocate() {
    m_interpreter->AllocateTensors();

    const auto numInputs = m_interpreter->inputs().size();

    auto& inputs = accessInputs();
    inputs.clear();
    inputs.reserve(numInputs);

    for (size_t i = 0; i < numInputs; ++i) {
        inputs.emplace_back(
            std::make_shared<TensorImpl>(m_interpreter->input_tensor(i)));
    }

    const auto numOutputs = m_interpreter->outputs().size();

    auto& outputs = accessOutputs();
    outputs.clear();
    outputs.reserve(numOutputs);

    for (size_t i = 0; i < numOutputs; ++i) {
        outputs.emplace_back(
            std::make_shared<TensorImpl>(m_interpreter->output_tensor(i)));
    }
}

auto ModelImpl::applyDelegate(const DELEGATE& delegate) -> STATUS {
    /* undo any previous delegate */
    createInterpreter();
    deleteDelegate();

    STATUS status = STATUS::SUCCESS;

    if (delegate == DELEGATE::GPU) {
#ifdef EDGERUNNER_GPU
        m_delegate = TfLiteGpuDelegateV2Create(nullptr);

        if (m_interpreter->ModifyGraphWithDelegate(m_delegate) != kTfLiteOk) {
            status = STATUS::FAIL;
        }

        setDelegate(delegate);
#else
        status = STATUS::FAIL;
#endif
    }

    allocate();

    return status;
}

auto ModelImpl::execute() -> STATUS {
    if (m_interpreter->Invoke() != kTfLiteOk) {
        return STATUS::FAIL;
    }

    return STATUS::SUCCESS;
}

void ModelImpl::deleteDelegate() {
    if (m_delegate != nullptr) {
#ifdef EDGERUNNER_GPU
        if (getDelegate() == DELEGATE::GPU) {
            TfLiteGpuDelegateV2Delete(m_delegate);
        }
#endif
    }
}

ModelImpl::~ModelImpl() {
    deleteDelegate();
}

}  // namespace edge::tflite
