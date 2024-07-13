#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>

#include "edgerunner/model.hpp"

#include <nonstd/span.hpp>
#include <tensorflow/lite/core/c/c_api_types.h>
#include <tensorflow/lite/interpreter_builder.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model_builder.h>

#include "edgerunner/tflite/model.hpp"
#include "edgerunner/tflite/tensor.hpp"

#ifdef EDGERUNNER_GPU
#    include <tensorflow/lite/delegates/gpu/delegate.h>
#endif

#ifdef EDGERUNNER_QNN
#    include <TFLiteDelegate/QnnTFLiteDelegate.h>
#endif

namespace edge::tflite {

ModelImpl::ModelImpl(const std::filesystem::path& modelPath)
    : Model(modelPath) {
    setCreationStatus(loadModel(modelPath));
    setCreationStatus(createInterpreter());
    setCreationStatus(allocate());
}

ModelImpl::ModelImpl(const nonstd::span<uint8_t>& modelBuffer) {
    setCreationStatus(loadModel(modelBuffer));
    setCreationStatus(createInterpreter());
    setCreationStatus(allocate());
}

auto ModelImpl::loadModel(const std::filesystem::path& modelPath) -> STATUS {
    m_modelBuffer = ::tflite::FlatBufferModel::BuildFromFile(modelPath.c_str());

    if (m_modelBuffer == nullptr) {
        return STATUS::FAIL;
    }

    return STATUS::SUCCESS;
}

auto ModelImpl::loadModel(const nonstd::span<uint8_t>& modelBuffer) -> STATUS {
    m_modelBuffer = ::tflite::FlatBufferModel::BuildFromBuffer(
        reinterpret_cast<char*> /* NOLINT */ (modelBuffer.data()),
        modelBuffer.size());

    if (m_modelBuffer == nullptr) {
        return STATUS::FAIL;
    }

    return STATUS::SUCCESS;
}

auto ModelImpl::createInterpreter() -> STATUS {
    const ::tflite::ops::builtin::BuiltinOpResolver opResolver;
    if (::tflite::InterpreterBuilder(*m_modelBuffer, opResolver)(&m_interpreter)
        != kTfLiteOk)
    {
        return STATUS::FAIL;
    }

    return STATUS::SUCCESS;
}

auto ModelImpl::allocate() -> STATUS {
    if (m_interpreter->AllocateTensors() != kTfLiteOk) {
        return STATUS::FAIL;
    }

    const auto numInputs = m_interpreter->inputs().size();

    auto& inputs = getInputs();
    inputs.clear();
    inputs.reserve(numInputs);

    for (size_t i = 0; i < numInputs; ++i) {
        inputs.emplace_back(
            std::make_shared<TensorImpl>(m_interpreter->input_tensor(i)));
    }

    const auto numOutputs = m_interpreter->outputs().size();

    auto& outputs = getOutputs();
    outputs.clear();
    outputs.reserve(numOutputs);

    for (size_t i = 0; i < numOutputs; ++i) {
        outputs.emplace_back(
            std::make_shared<TensorImpl>(m_interpreter->output_tensor(i)));
    }

    return STATUS::SUCCESS;
}

auto ModelImpl::applyDelegate(const DELEGATE& delegate) -> STATUS {
    /* undo any previous delegate */
    if (createInterpreter() != STATUS::SUCCESS) {
        return STATUS::FAIL;
    }

    deleteDelegate();

    STATUS status = STATUS::SUCCESS;
    if (delegate == DELEGATE::CPU) {
        setDelegate(delegate);
    } else if (delegate == DELEGATE::GPU) {
#ifdef EDGERUNNER_GPU
        m_delegate = TfLiteGpuDelegateV2Create(nullptr);

        if (m_interpreter->ModifyGraphWithDelegate(m_delegate) != kTfLiteOk) {
            status = STATUS::FAIL;
            setDelegate(DELEGATE::CPU);
        } else {
            setDelegate(delegate);
        }
#else
        status = STATUS::FAIL;
#endif
    } else if (delegate == DELEGATE::NPU) {
#ifdef EDGERUNNER_QNN
        TfLiteQnnDelegateOptions options = TfLiteQnnDelegateOptionsDefault();

        options.backend_type = kHtpBackend;
        options.log_level = kLogOff;
        options.htp_options.precision = kHtpFp16;
        options.htp_options.performance_mode = kHtpBurst;

        m_delegate = TfLiteQnnDelegateCreate(&options);

        if (m_interpreter->ModifyGraphWithDelegate(m_delegate) != kTfLiteOk) {
            status = STATUS::FAIL;
            setDelegate(DELEGATE::CPU);
        } else {
            setDelegate(delegate);
        }
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

        if (getDelegate() == DELEGATE::NPU) {
#ifdef EDGERUNNER_QNN
            TfLiteQnnDelegateDelete(m_delegate);
#endif
        }
    }
}

ModelImpl::~ModelImpl() {
    deleteDelegate();
}

}  // namespace edge::tflite
