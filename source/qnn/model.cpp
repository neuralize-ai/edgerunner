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

#include "edgerunner/qnn/config.h"
#include "edgerunner/qnn/model.hpp"
#include "edgerunner/qnn/tensor.hpp"

namespace edge::qnn {

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

    auto& graphInfo = (*m_graphInfo)[0];

    nonstd::span<Qnn_Tensor_t> inputTensorSpecs {graphInfo.inputTensors,
                                                 graphInfo.numInputTensors};
    nonstd::span<Qnn_Tensor_t> outputTensorSpecs {graphInfo.outputTensors,
                                                  graphInfo.numOutputTensors};

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
    auto& graphInfo = (*m_graphInfo)[0];

    const auto executeStatus =
        qnnInterface.graphExecute(graphInfo.graph,
                                  graphInfo.inputTensors,
                                  graphInfo.numInputTensors,
                                  graphInfo.outputTensors,
                                  graphInfo.numOutputTensors,
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
        auto& graphCustomConfig = graphConfigs.createCustomConfig();
        graphCustomConfig.option = QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION;
        graphCustomConfig.precision = QNN_PRECISION_FLOAT16;

        auto& graphConfig = graphConfigs.createConfig();
        graphConfig.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
        graphConfig.customConfig = &graphCustomConfig;
    }

    const auto status = qnnInterface.graphSetConfig((*m_graphInfo)[0].graph,
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

    const auto status = m_composeGraphsFnHandle(qnnBackendHandle,
                                                qnnInterface,
                                                qnnContext,
                                                nullptr,
                                                0,
                                                &m_graphInfo,
                                                &m_graphsCount,
                                                false,
                                                nullptr,
                                                QNN_LOG_LEVEL_ERROR);

    if (ModelErrorT::MODEL_NO_ERROR != status) {
        return STATUS::FAIL;
    }

    return STATUS::SUCCESS;
}

auto ModelImpl::finalizeGraphs() -> STATUS {
    auto& qnnInterface = m_backend->getInterface();
    auto& graphInfo = (*m_graphInfo)[0];

    const auto status =
        qnnInterface.graphFinalize(graphInfo.graph, nullptr, nullptr);

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

    m_composeGraphsFnHandle = reinterpret_cast<ComposeGraphsFnHandleTypeT>(
        dlsym(m_libModelHandle, "QnnModel_composeGraphs"));
    if (nullptr == m_composeGraphsFnHandle) {
        return STATUS::FAIL;
    }

    m_freeGraphInfoFnHandle = reinterpret_cast<FreeGraphInfoFnHandleTypeT>(
        dlsym(m_libModelHandle, "QnnModel_freeGraphsInfo"));
    if (nullptr == m_freeGraphInfoFnHandle) {
        return STATUS::FAIL;
    }

    return STATUS::SUCCESS;
}

}  // namespace edge::qnn
