#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <ios>
#include <memory>
#include <string>
#include <vector>

#include "edgerunner/model.hpp"

#include <HTP/QnnHtpGraph.h>
#include <QnnContext.h>
#include <QnnGraph.h>
#include <QnnInterface.h>
#include <QnnLog.h>
#include <QnnTypes.h>
#include <System/QnnSystemContext.h>
#include <dlfcn.h>
#include <nonstd/span.hpp>

#include "edgerunner/qnn/backend.hpp"
#include "edgerunner/qnn/config.hpp"
#include "edgerunner/qnn/model.hpp"
#include "edgerunner/qnn/tensor.hpp"
#include "edgerunner/tensor.hpp"

namespace edge::qnn {

ModelImpl::ModelImpl(const std::filesystem::path& modelPath)
    : Model(modelPath) {
    const auto modelExtension = modelPath.extension().string().substr(1);
    m_loadCachedBinary = modelExtension == "bin";

    m_backend = std::make_unique<Backend>(DELEGATE::NPU, m_loadCachedBinary);

    if (!m_loadCachedBinary) {
        setCreationStatus(loadModel(modelPath));
        setCreationStatus(composeGraphs());
        setPrecision(detectPrecision());
        setCreationStatus(setGraphConfig());
        setCreationStatus(finalizeGraphs());
    } else {
        std::ifstream file(modelPath, std::ios::binary);
        if (!file) {
            setCreationStatus(STATUS::FAIL);
            return;
        }

        const auto bufferSize = std::filesystem::file_size(modelPath);

        std::vector<uint8_t> modelBuffer(bufferSize);

        if (!file.read(
                reinterpret_cast<char*> /* NOLINT */ (modelBuffer.data()),
                static_cast<std::streamsize>(modelBuffer.size())))
        {
            setCreationStatus(STATUS::FAIL);
            return;
        }

        setCreationStatus(loadModel(modelBuffer));
    }

    setCreationStatus(allocate());
}

ModelImpl::ModelImpl(const nonstd::span<uint8_t>& modelBuffer) {
    setCreationStatus(loadModel(modelBuffer));
}

auto ModelImpl::loadModel(const std::filesystem::path& modelPath) -> STATUS {
    return loadFromSharedLibrary(modelPath);
}

auto ModelImpl::loadModel(const nonstd::span<uint8_t>& modelBuffer) -> STATUS {
    return loadFromContextBinary(modelBuffer);
}

auto ModelImpl::applyDelegate(const DELEGATE& delegate) -> STATUS {
    if (delegate != DELEGATE::NPU) {
        return STATUS::FAIL;
    }
    return STATUS::SUCCESS;
}

auto ModelImpl::detectPrecision() -> TensorType {
    const auto inputTensorSpecs = m_graphInfo.getInputs();

    std::vector<TensorImpl> inputs;
    inputs.reserve(inputTensorSpecs.size());
    for (auto& inputTensorSpec : inputTensorSpecs) {
        inputs.emplace_back(&inputTensorSpec, false);
    }

    for (auto& input : inputs) {
        const auto type = input.getType();

        if (type == TensorType::FLOAT16 || type == TensorType::FLOAT32) {
            return TensorType::FLOAT16;
        }
    }

    return TensorType::UINT8;
}

auto ModelImpl::allocate() -> STATUS {
    auto& inputs = getInputs();
    auto& outputs = getOutputs();

    inputs.clear();
    outputs.clear();

    const auto inputTensorSpecs = m_graphInfo.getInputs();
    const auto outputTensorSpecs = m_graphInfo.getOutputs();

    if (inputTensorSpecs.data() == nullptr
        || outputTensorSpecs.data() == nullptr)
    {
        return STATUS::FAIL;
    }

    inputs.reserve(inputTensorSpecs.size());
    for (auto& inputTensorSpec : inputTensorSpecs) {
        inputs.emplace_back(std::make_shared<TensorImpl>(&inputTensorSpec));
    }

    outputs.reserve(outputTensorSpecs.size());
    for (auto& outputTensorSpec : outputTensorSpecs) {
        outputs.emplace_back(std::make_shared<TensorImpl>(&outputTensorSpec));
    }

    return STATUS::SUCCESS;
}

auto ModelImpl::execute() -> STATUS {
    auto& qnnInterface = m_backend->getInterface();

    const auto executeStatus =
        qnnInterface.graphExecute(m_graphInfo.getGraph(),
                                  m_graphInfo.getInputs().data(),
                                  m_graphInfo.getNumInputs(),
                                  m_graphInfo.getOutputs().data(),
                                  m_graphInfo.getNumOutputs(),
                                  nullptr,
                                  nullptr);
    if (QNN_GRAPH_NO_ERROR != executeStatus) {
        return STATUS::FAIL;
    }

    return STATUS::SUCCESS;
}

auto ModelImpl::setGraphConfig() -> STATUS {
    auto& qnnInterface = m_backend->getInterface();

    Config<QnnGraph_Config_t, QnnHtpGraph_CustomConfig_t> graphConfigs {
        QNN_GRAPH_CONFIG_INIT, QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT};

    if (m_backend->getDelegate() == DELEGATE::NPU) {
        if (getPrecision() == TensorType::FLOAT16) {
            auto& precisionCustomConfig = graphConfigs.createCustomConfig();
            precisionCustomConfig.option =
                QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION;

            precisionCustomConfig.precision /* NOLINT */ =
                QNN_PRECISION_FLOAT16;

            auto& precisionConfig = graphConfigs.createConfig();
            precisionConfig.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
            precisionConfig.customConfig /* NOLINT */ = &precisionCustomConfig;
        }

        auto& optimizationCustomConfig = graphConfigs.createCustomConfig();
        optimizationCustomConfig.option =
            QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION;
        optimizationCustomConfig.optimizationOption /* NOLINT */.type =
            QNN_HTP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_OPTIMIZATION_FLAG;
        static constexpr float GraphOptimizationLevel = 3.0F;
        optimizationCustomConfig.optimizationOption /* NOLINT */.floatValue =
            GraphOptimizationLevel;

        auto& optimizationConfig = graphConfigs.createConfig();
        optimizationConfig.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
        optimizationConfig.customConfig /* NOLINT */ =
            &optimizationCustomConfig;
    }

    const auto status = qnnInterface.graphSetConfig(m_graphInfo.getGraph(),
                                                    graphConfigs.getPtr());

    if (QNN_GRAPH_NO_ERROR != status) {
        return STATUS::FAIL;
    }

    return STATUS::SUCCESS;
}

auto ModelImpl::composeGraphs() -> STATUS {
    auto& qnnInterface = m_backend->getInterface();
    auto& qnnContext = m_backend->getContext();
    auto& qnnBackendHandle = m_backend->getHandle();

    const auto status =
        m_graphInfo.composeGraphs(qnnBackendHandle, qnnInterface, qnnContext);

    if (GraphErrorT::GRAPH_NO_ERROR != status) {
        return STATUS::FAIL;
    }

    return STATUS::SUCCESS;
}

auto ModelImpl::finalizeGraphs() -> STATUS {
    auto& qnnInterface = m_backend->getInterface();

    const auto status =
        qnnInterface.graphFinalize(m_graphInfo.getGraph(), nullptr, nullptr);

    if (QNN_GRAPH_NO_ERROR != status) {
        return STATUS::FAIL;
    }

    saveContextBinary(name() + ".bin");

    return STATUS::SUCCESS;
}

auto ModelImpl::saveContextBinary(const std::filesystem::path& binaryPath)
    -> STATUS {
    auto& qnnInterface = m_backend->getInterface();
    auto& context = m_backend->getContext();

    if (nullptr == qnnInterface.contextGetBinarySize
        || nullptr == qnnInterface.contextGetBinary)
    {
        return STATUS::FAIL;
    }
    uint64_t requiredBufferSize {0};
    if (QNN_CONTEXT_NO_ERROR
        != qnnInterface.contextGetBinarySize(context, &requiredBufferSize))
    {
        return STATUS::FAIL;
    }

    std::vector<uint8_t> buffer(requiredBufferSize);

    uint64_t writtenBufferSize {0};
    if (QNN_CONTEXT_NO_ERROR
        != qnnInterface.contextGetBinary(
            context,
            reinterpret_cast<void*> /* NOLINT */ (buffer.data()),
            requiredBufferSize,
            &writtenBufferSize))
    {
        return STATUS::FAIL;
    }

    if (requiredBufferSize < writtenBufferSize) {
        return STATUS::FAIL;
    }

    std::ofstream file(binaryPath, std::ofstream::binary);
    file.write(reinterpret_cast<const char*> /* NOLINT */ (buffer.data()),
               static_cast<std::streamsize>(buffer.size()));

    return STATUS::SUCCESS;
}

auto ModelImpl::loadFromSharedLibrary(const std::filesystem::path& modelPath)
    -> STATUS {
    return m_graphInfo.loadFromSharedLibrary(modelPath);
}

auto ModelImpl::loadFromContextBinary(const nonstd::span<uint8_t>& modelBuffer)
    -> STATUS {
    auto& qnnSystemInterface = m_backend->getSystemInterface();
    QnnSystemContext_Handle_t sysCtxHandle {nullptr};
    if (QNN_SUCCESS != qnnSystemInterface.systemContextCreate(&sysCtxHandle)) {
        return STATUS::FAIL;
    }
    const QnnSystemContext_BinaryInfo_t* binaryInfo {nullptr};
    Qnn_ContextBinarySize_t binaryInfoSize {0};
    if (QNN_SUCCESS
        != qnnSystemInterface.systemContextGetBinaryInfo(
            sysCtxHandle,
            static_cast<void*>(modelBuffer.data()),
            modelBuffer.size(),
            &binaryInfo,
            &binaryInfoSize))
    {
        return STATUS::FAIL;
    }

    if (!m_graphInfo.copyMetadataToGraphsInfo(binaryInfo)) {
        return STATUS::FAIL;
    }

    qnnSystemInterface.systemContextFree(sysCtxHandle);
    sysCtxHandle = nullptr;

    auto& qnnInterface = m_backend->getInterface();

    if (nullptr == qnnInterface.contextCreateFromBinary) {
        return STATUS::FAIL;
    }

    auto& backendHandle = m_backend->getHandle();
    auto& deviceHandle = m_backend->getDeviceHandle();
    auto& context = m_backend->getContext();

    Config<QnnContext_Config_t, void*> contextConfig {QNN_CONTEXT_CONFIG_INIT,
                                                      {}};
    if (qnnInterface.contextCreateFromBinary(
            backendHandle,
            deviceHandle,
            contextConfig.getPtr(),
            static_cast<void*>(modelBuffer.data()),
            modelBuffer.size(),
            &context,
            nullptr)
        != 0U)
    {
        return STATUS::FAIL;
    }

    return m_graphInfo.retrieveGraphFromContext(qnnInterface, context);
}

}  // namespace edge::qnn
