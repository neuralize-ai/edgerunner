#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>

#include "edgerunner/model.hpp"

#include <HTP/QnnHtpGraph.h>
#include <QnnGraph.h>
#include <QnnInterface.h>
#include <QnnLog.h>
#include <QnnTypes.h>
#include <dlfcn.h>
#include <nonstd/span.hpp>

#include "edgerunner/qnn/backend.hpp"
#include "edgerunner/qnn/config.hpp"
#include "edgerunner/qnn/model.hpp"
#include "edgerunner/qnn/tensor.hpp"
#include "edgerunner/qnn/tensorOps.hpp"
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

ModelImpl::~ModelImpl() {
    if (m_graphsInfo != nullptr && m_freeGraphInfoFnHandle != nullptr) {
        m_freeGraphInfoFnHandle(&m_graphsInfo, m_graphsCount);
    }

    if (m_libModelHandle != nullptr) {
        dlclose(m_libModelHandle);
    }
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
    nonstd::span<Qnn_Tensor_t> inputTensorSpecs {m_graphInfo->inputTensors,
                                                 m_graphInfo->numInputTensors};

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

auto deepCopyQnnTensorInfo(Qnn_Tensor_t& dst, const Qnn_Tensor_t& src) -> bool {
    dst.version = src.version;
    const char* tensorName = getQnnTensorName(src);
    if (tensorName == nullptr) {
        setQnnTensorName(dst, nullptr);
    } else {
        setQnnTensorName(dst, strndup(tensorName, strlen(tensorName)));
    }
    setQnnTensorId(dst, getQnnTensorId(src));
    setQnnTensorType(dst, getQnnTensorType(src));
    setQnnTensorDataFormat(dst, getQnnTensorDataFormat(src));
    setQnnTensorDataType(dst, getQnnTensorDataType(src));
    Qnn_QuantizeParams_t qParams = QNN_QUANTIZE_PARAMS_INIT;
    qParams.encodingDefinition =
        getQnnTensorQuantParams(src).encodingDefinition;
    qParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
    if (getQnnTensorQuantParams(src).quantizationEncoding
        == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET)
    {
        qParams.quantizationEncoding =
            getQnnTensorQuantParams(src).quantizationEncoding;
        qParams.scaleOffsetEncoding /* NOLINT */ =
            getQnnTensorQuantParams(src).scaleOffsetEncoding /* NOLINT */;
    } else if (getQnnTensorQuantParams(src).quantizationEncoding
               == QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET)
    {
        qParams.quantizationEncoding =
            getQnnTensorQuantParams(src).quantizationEncoding;
        qParams.axisScaleOffsetEncoding /* NOLINT */.axis =
            getQnnTensorQuantParams(src)
                .axisScaleOffsetEncoding /* NOLINT */.axis;
        qParams.axisScaleOffsetEncoding /* NOLINT */.numScaleOffsets =
            getQnnTensorQuantParams(src)
                .axisScaleOffsetEncoding /* NOLINT */.numScaleOffsets;
        if (getQnnTensorQuantParams(src)
                .axisScaleOffsetEncoding /* NOLINT */.numScaleOffsets
            > 0)
        {
            qParams.axisScaleOffsetEncoding.scaleOffset /* NOLINT */ =
                reinterpret_cast<Qnn_ScaleOffset_t*> /* NOLINT */ (malloc(
                    getQnnTensorQuantParams(src)
                        .axisScaleOffsetEncoding /* NOLINT */.numScaleOffsets
                    * sizeof(Qnn_ScaleOffset_t)));
            if (qParams.axisScaleOffsetEncoding /* NOLINT */.scaleOffset
                != nullptr)
            {
                for (size_t idx = 0;
                     idx < getQnnTensorQuantParams(src)
                               .axisScaleOffsetEncoding /* NOLINT */
                               .numScaleOffsets;
                     idx++)
                {
                    qParams
                        .axisScaleOffsetEncoding /* NOLINT */.scaleOffset[idx]
                        .scale = getQnnTensorQuantParams(src) /* NOLINT */
                                     .axisScaleOffsetEncoding /* NOLINT */
                                     .scaleOffset[idx]
                                     .scale;
                    qParams
                        .axisScaleOffsetEncoding /* NOLINT */.scaleOffset[idx]
                        .offset = getQnnTensorQuantParams(src) /* NOLINT */
                                      .axisScaleOffsetEncoding /* NOLINT */
                                      .scaleOffset[idx]
                                      .offset;
                }
            }
        }
    }
    setQnnTensorQuantParams(dst, qParams);
    setQnnTensorRank(dst, getQnnTensorRank(src));
    setQnnTensorDimensions(dst, nullptr);
    if (getQnnTensorRank(src) > 0) {
        setQnnTensorDimensions(dst,
                               static_cast<uint32_t*>(malloc /* NOLINT */ (
                                   getQnnTensorRank(src) * sizeof(uint32_t))));
        if (getQnnTensorDimensions(dst) != nullptr) {
            memcpy(getQnnTensorDimensions(dst),
                   getQnnTensorDimensions(src),
                   getQnnTensorRank(src) * sizeof(uint32_t));
        }
        if (getQnnTensorIsDynamicDimensions(src) != nullptr) {
            setQnnTensorIsDynamicDimensions(
                dst,
                static_cast<uint8_t*>(malloc /* NOLINT */ (getQnnTensorRank(src)
                                                           * sizeof(uint8_t))));
            memcpy(getQnnTensorIsDynamicDimensions(dst),
                   getQnnTensorIsDynamicDimensions(src),
                   getQnnTensorRank(src) * sizeof(uint8_t));
        }
    }

    setQnnTensorSparseParams(dst, getQnnTensorSparseParams(src));

    return true;
}

auto copyTensorsInfo(const Qnn_Tensor_t* tensorsInfoSrc,
                     Qnn_Tensor_t*& tensorWrappers,
                     uint32_t tensorsCount) -> bool {
    tensorWrappers /* NOLINT */ = static_cast<Qnn_Tensor_t*>(
        calloc /* NOLINT */ (tensorsCount, sizeof(Qnn_Tensor_t)));
    if (nullptr == tensorWrappers) {
        return false;
    }
    for (size_t tIdx = 0; tIdx < tensorsCount; ++tIdx) {
        tensorWrappers[tIdx] /* NOLINT */ = QNN_TENSOR_INIT;
        if (!deepCopyQnnTensorInfo(tensorWrappers[tIdx], tensorsInfoSrc[tIdx]))
        {
            return false;
        }
    }

    return true;
}

auto copyGraphsInfoV1(const QnnSystemContext_GraphInfoV1_t* graphInfoSrc,
                      GraphInfoT* graphInfoDst) -> bool {
    graphInfoDst->graphName = nullptr;
    if (graphInfoSrc->graphName != nullptr) {
        graphInfoDst->graphName =
            strndup(graphInfoSrc->graphName, strlen(graphInfoSrc->graphName));
    }
    graphInfoDst->inputTensors = nullptr;
    graphInfoDst->numInputTensors = 0;
    if (graphInfoSrc->graphInputs != nullptr) {
        if (!copyTensorsInfo(graphInfoSrc->graphInputs,
                             graphInfoDst->inputTensors,
                             graphInfoSrc->numGraphInputs))
        {
            return false;
        }
        graphInfoDst->numInputTensors = graphInfoSrc->numGraphInputs;
    }
    graphInfoDst->outputTensors = nullptr;
    graphInfoDst->numOutputTensors = 0;
    if (graphInfoSrc->graphOutputs != nullptr) {
        if (!copyTensorsInfo(graphInfoSrc->graphOutputs,
                             graphInfoDst->outputTensors,
                             graphInfoSrc->numGraphOutputs))
        {
            return false;
        }
        graphInfoDst->numOutputTensors = graphInfoSrc->numGraphOutputs;
    }
    return true;
}

auto copyGraphsInfo(const QnnSystemContext_GraphInfo_t* graphsInput,
                    const uint32_t numGraphs,
                    GraphInfoT**& graphsInfo) -> bool {
    if (graphsInput == nullptr) {
        return false;
    }
    graphsInfo =
        static_cast<GraphInfoT**>(calloc(numGraphs, sizeof(GraphInfoT*)));
    auto* graphInfoArr =
        static_cast<GraphInfoT*>(calloc(numGraphs, sizeof(GraphInfoT)));
    if (nullptr == graphsInfo || nullptr == graphInfoArr) {
        return false;
    }

    for (size_t gIdx = 0; gIdx < numGraphs; ++gIdx) {
        if (graphsInput[gIdx].version
            == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1)
        {
            if (!copyGraphsInfoV1(&graphsInput[gIdx].graphInfoV1,
                                  &graphInfoArr[gIdx]))
            {
                return false;
            }
        }
        graphsInfo[gIdx] = graphInfoArr + gIdx;
    }

    return true;
}

auto copyMetadataToGraphsInfo(const QnnSystemContext_BinaryInfo_t* binaryInfo,
                              GraphInfoT**& graphsInfo,
                              uint32_t& graphsCount) -> bool {
    if (nullptr == binaryInfo) {
        return false;
    }
    graphsCount = 0;
    if (binaryInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
        if (binaryInfo->contextBinaryInfoV1.graphs != nullptr) {
            if (!copyGraphsInfo(binaryInfo->contextBinaryInfoV1.graphs,
                                binaryInfo->contextBinaryInfoV1.numGraphs,
                                graphsInfo))
            {
                return false;
            }
            graphsCount = binaryInfo->contextBinaryInfoV1.numGraphs;
            return true;
        }
    } else if (binaryInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2)
    {
        if (binaryInfo->contextBinaryInfoV2.graphs != nullptr) {
            if (!copyGraphsInfo(binaryInfo->contextBinaryInfoV2.graphs,
                                binaryInfo->contextBinaryInfoV2.numGraphs,
                                graphsInfo))
            {
                return false;
            }
            graphsCount = binaryInfo->contextBinaryInfoV2.numGraphs;
            return true;
        }
    }
    return false;
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

    if (!copyMetadataToGraphsInfo(binaryInfo, m_graphsInfo, m_graphsCount)) {
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

    for (size_t graphIdx = 0; graphIdx < m_graphsCount; ++graphIdx) {
        if (nullptr == qnnInterface.graphRetrieve) {
            return STATUS::FAIL;
        }
        auto& graphInfo = (*m_graphsInfo) /* NOLINT */[graphIdx];
        if (QNN_SUCCESS
            != qnnInterface.graphRetrieve(
                context, graphInfo.graphName, &graphInfo.graph))
        {
            return STATUS::FAIL;
        }
    }

    m_graphInfo = m_graphsInfo[0];  // NOLINT

    return STATUS::SUCCESS;
}

}  // namespace edge::qnn
