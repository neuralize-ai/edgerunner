#include "edgerunner/qnn/graph.hpp"

namespace edge::qnn {

GraphsInfo::~GraphsInfo() {
    if (m_graphsInfo == nullptr) {
        return;
    }

    if (m_freeGraphInfoFnHandle != nullptr) {
        m_freeGraphInfoFnHandle(&m_graphsInfo, m_graphsCount);
        return;
    }
    if (m_libModelHandle != nullptr) {
        dlclose(m_libModelHandle);
    }
}

auto GraphsInfo::loadFromSharedLibrary(const std::filesystem::path& modelPath)
    -> STATUS {
    m_libModelHandle = dlopen(modelPath.string().data(), RTLD_NOW | RTLD_LOCAL);

    if (nullptr == m_libModelHandle) {
        return STATUS::FAIL;
    }

    auto status = setComposeGraphsFnHandle(
        reinterpret_cast<ComposeGraphsFnHandleTypeT> /* NOLINT */ (
            dlsym(m_libModelHandle, "QnnModel_composeGraphs")));

    if (status == STATUS::FAIL) {
        return status;
    }

    status = setFreeGraphInfoFnHandle(
        reinterpret_cast<FreeGraphInfoFnHandleTypeT> /* NOLINT */ (
            dlsym(m_libModelHandle, "QnnModel_freeGraphsInfo")));

    return status;
}

auto GraphsInfo::setComposeGraphsFnHandle(
    ComposeGraphsFnHandleTypeT composeGraphsFnHandle) -> STATUS {
    m_composeGraphsFnHandle = composeGraphsFnHandle;

    if (m_composeGraphsFnHandle == nullptr) {
        return STATUS::FAIL;
    }

    return STATUS::SUCCESS;
}

auto GraphsInfo::setFreeGraphInfoFnHandle(
    FreeGraphInfoFnHandleTypeT freeGraphInfoFnHandle) -> STATUS {
    m_freeGraphInfoFnHandle = freeGraphInfoFnHandle;

    if (m_freeGraphInfoFnHandle == nullptr) {
        return STATUS::FAIL;
    }

    return STATUS::SUCCESS;
}

auto GraphsInfo::composeGraphs(Qnn_BackendHandle_t& qnnBackendHandle,
                               QnnInterface_ImplementationV2_16_t& qnnInterface,
                               Qnn_ContextHandle_t& qnnContext) -> GraphErrorT {
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

    setGraph();

    return status;
}

auto GraphsInfo::retrieveGraphFromContext(
    QnnInterface_ImplementationV2_16_t& qnnInterface,
    Qnn_ContextHandle_t& qnnContext) -> STATUS {
    for (size_t graphIdx = 0; graphIdx < m_graphsCount; ++graphIdx) {
        if (nullptr == qnnInterface.graphRetrieve) {
            return STATUS::FAIL;
        }
        auto& graphInfo = (*m_graphsInfo)[graphIdx] /* NOLINT */;
        if (QNN_SUCCESS
            != qnnInterface.graphRetrieve(
                qnnContext, graphInfo.graphName, &graphInfo.graph))
        {
            return STATUS::FAIL;
        }
    }

    setGraph();

    return STATUS::SUCCESS;
}

auto GraphsInfo::copyGraphsInfoV1(
    const QnnSystemContext_GraphInfoV1_t* graphInfoSrc,
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

auto GraphsInfo::copyGraphsInfo(const QnnSystemContext_GraphInfo_t* graphsInput,
                                const uint32_t numGraphs) -> bool {
    if (graphsInput == nullptr) {
        return false;
    }
    m_graphs.resize(numGraphs);
    m_graphPtrs.clear();
    m_graphPtrs.reserve(numGraphs);

    for (auto& graph : m_graphs) {
        m_graphPtrs.push_back(&graph);
    }

    m_graphsInfo = m_graphPtrs.data();

    nonstd::span<const QnnSystemContext_GraphInfo_t> srcGraphs {graphsInput,
                                                                numGraphs};

    for (uint32_t i = 0; i < numGraphs; ++i) {
        if (srcGraphs[i].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
            if (!copyGraphsInfoV1(&srcGraphs[i].graphInfoV1 /* NOLINT */,
                                  m_graphPtrs[i]))
            {
                return false;
            }
        }
    }

    return true;
}

auto GraphsInfo::copyMetadataToGraphsInfo(
    const QnnSystemContext_BinaryInfo_t* binaryInfo) -> bool {
    if (nullptr == binaryInfo) {
        return false;
    }

    m_graphsCount = 0;
    auto binaryInfoVariant = getContextBinaryInfoVariant(binaryInfo);

    return std::visit(
        [this](auto&& typedBinaryInfo) {
            if (!copyGraphsInfo(typedBinaryInfo.get().graphs,
                                typedBinaryInfo.get().numGraphs))
            {
                return false;
            }
            m_graphsCount = typedBinaryInfo.get().numGraphs;
            return true;
        },
        binaryInfoVariant

    );
}

}  // namespace edge::qnn
