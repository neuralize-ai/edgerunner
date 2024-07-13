#include <cstdint>
#include <filesystem>
#include <memory>

#include "edgerunner/model.hpp"

#include <HTP/QnnHtpGraph.h>
#include <QnnGraph.h>
#include <QnnInterface.h>
#include <QnnLog.h>
#include <QnnTypes.h>
#include <dlfcn.h>
#include <nonstd/span.hpp>

#include "edgerunner/qnn/backend.h"
#include "edgerunner/qnn/config.h"
#include "edgerunner/qnn/model.hpp"
#include "edgerunner/qnn/tensor.hpp"

namespace edge::qnn {

ModelImpl::ModelImpl(const std::filesystem::path& modelPath)
    : Model(modelPath) {
    const auto modelExtension = modelPath.extension().string().substr(1);
    m_loadCachedBinary = modelExtension == "bin";

    m_backend = std::make_unique<Backend>(DELEGATE::NPU);

    setCreationStatus(loadModel(modelPath));

    if (!m_loadCachedBinary) {
        setCreationStatus(composeGraphs());
        setCreationStatus(setGraphConfig());
        setCreationStatus(finalizeGraphs());
    }

    setCreationStatus(allocate());
}

ModelImpl::ModelImpl(const nonstd::span<uint8_t>& modelBuffer) {
    setCreationStatus(loadModel(modelBuffer));
}

auto ModelImpl::loadModel(const std::filesystem::path& modelPath) -> STATUS {
    return loadFromSharedLibrary(modelPath);
}

auto ModelImpl::loadModel(const nonstd::span<uint8_t>& /*modelBuffer*/)
    -> STATUS {
    return STATUS::FAIL;
}

auto ModelImpl::applyDelegate(const DELEGATE& delegate) -> STATUS {
    if (delegate != DELEGATE::NPU) {
        return STATUS::FAIL;
    }
    return STATUS::SUCCESS;
}

auto ModelImpl::allocate() -> STATUS {
    auto& inputs = getInputs();
    auto& outputs = getOutputs();

    inputs.clear();
    outputs.clear();

    nonstd::span<Qnn_Tensor_t> inputTensorSpecs {m_graphInfo->inputTensors,
                                                 m_graphInfo->numInputTensors};
    nonstd::span<Qnn_Tensor_t> outputTensorSpecs {
        m_graphInfo->outputTensors, m_graphInfo->numOutputTensors};

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
        qnnInterface.graphExecute(m_graphInfo->graph,
                                  m_graphInfo->inputTensors,
                                  m_graphInfo->numInputTensors,
                                  m_graphInfo->outputTensors,
                                  m_graphInfo->numOutputTensors,
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

    /* TODO: determine desired precision */
    if (m_backend->getDelegate() == DELEGATE::NPU) {
        auto& precisionCustomConfig = graphConfigs.createCustomConfig();
        precisionCustomConfig.option = QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION;
        precisionCustomConfig.precision /* NOLINT */ = QNN_PRECISION_FLOAT16;

        auto& precisionConfig = graphConfigs.createConfig();
        precisionConfig.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
        precisionConfig.customConfig /* NOLINT */ = &precisionCustomConfig;

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

    const auto status =
        qnnInterface.graphSetConfig(m_graphInfo->graph, graphConfigs.getPtr());

    if (QNN_GRAPH_NO_ERROR != status) {
        return STATUS::FAIL;
    }

    return STATUS::SUCCESS;
}

auto ModelImpl::composeGraphs() -> STATUS {
    auto& qnnInterface = m_backend->getInterface();
    auto& qnnContext = m_backend->getContext();
    auto& qnnBackendHandle = m_backend->getHandle();

    const auto status = m_composeGraphsFnHandle(qnnBackendHandle,
                                                qnnInterface,
                                                qnnContext,
                                                nullptr,
                                                0,
                                                &m_graphsInfo,
                                                &m_graphsCount,
                                                false,
                                                nullptr,
                                                QNN_LOG_LEVEL_ERROR);

    if (ModelErrorT::MODEL_NO_ERROR != status) {
        return STATUS::FAIL;
    }

    m_graphInfo = m_graphsInfo[0];  // NOLINT

    return STATUS::SUCCESS;
}

auto ModelImpl::finalizeGraphs() -> STATUS {
    auto& qnnInterface = m_backend->getInterface();

    const auto status =
        qnnInterface.graphFinalize(m_graphInfo->graph, nullptr, nullptr);

    if (QNN_GRAPH_NO_ERROR != status) {
        return STATUS::FAIL;
    }

    /* TODO: save binary */

    return STATUS::SUCCESS;
}

auto ModelImpl::loadFromSharedLibrary(const std::filesystem::path& modelPath)
    -> STATUS {
    m_libModelHandle = dlopen(modelPath.string().data(), RTLD_NOW | RTLD_LOCAL);

    if (nullptr == m_libModelHandle) {
        return STATUS::FAIL;
    }

    m_composeGraphsFnHandle =
        reinterpret_cast<ComposeGraphsFnHandleTypeT> /* NOLINT */ (
            dlsym(m_libModelHandle, "QnnModel_composeGraphs"));
    if (nullptr == m_composeGraphsFnHandle) {
        return STATUS::FAIL;
    }

    m_freeGraphInfoFnHandle =
        reinterpret_cast<FreeGraphInfoFnHandleTypeT> /* NOLINT */ (
            dlsym(m_libModelHandle, "QnnModel_freeGraphsInfo"));
    if (nullptr == m_freeGraphInfoFnHandle) {
        return STATUS::FAIL;
    }

    return STATUS::SUCCESS;
}

}  // namespace edge::qnn
