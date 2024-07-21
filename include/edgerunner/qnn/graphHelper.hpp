#pragma once

#include <cstring>
#include <memory>

#include <QnnCommon.h>
#include <QnnGraph.h>
#include <QnnInterface.h>
#include <QnnTypes.h>
#include <System/QnnSystemContext.h>
#include <dlfcn.h>
#include <nonstd/span.hpp>

#include "edgerunner/model.hpp"
#include "edgerunner/qnn/tensorOps.hpp"

namespace edge::qnn {

using GraphErrorT = enum GraphError {
    GRAPH_NO_ERROR = 0,
    GRAPH_TENSOR_ERROR = 1,
    GRAPH_PARAMS_ERROR = 2,
    GRAPH_NODES_ERROR = 3,
    GRAPH_GRAPH_ERROR = 4,
    GRAPH_CONTEXT_ERROR = 5,
    GRAPH_GENERATION_ERROR = 6,
    GRAPH_SETUP_ERROR = 7,
    GRAPH_INVALID_ARGUMENT_ERROR = 8,
    GRAPH_FILE_ERROR = 9,
    GRAPH_MEMORY_ALLOCATE_ERROR = 10,
    // Value selected to ensure 32 bits.
    GRAPH_UNKNOWN_ERROR = 0x7FFFFFFF
};

using GraphInfoT = struct GraphInfo {
    Qnn_GraphHandle_t graph;
    char* graphName;
    Qnn_Tensor_t* inputTensors;
    uint32_t numInputTensors;
    Qnn_Tensor_t* outputTensors;
    uint32_t numOutputTensors;
};

using GraphConfigInfoT = struct GraphConfigInfo {
    char* graphName;
    const QnnGraph_Config_t** graphConfigs;
};

using ComposeGraphsFnHandleTypeT =
    GraphErrorT (*)(Qnn_BackendHandle_t,
                    QnnInterface_ImplementationV2_16_t,
                    Qnn_ContextHandle_t,
                    const GraphConfigInfoT**,
                    const uint32_t,
                    GraphInfoT***,
                    uint32_t*,
                    bool,
                    QnnLog_Callback_t,
                    QnnLog_Level_t);

using FreeGraphInfoFnHandleTypeT = GraphErrorT (*)(GraphInfoT***, uint32_t);

class GraphInfoWrapper {
  public:
    GraphInfoWrapper() = default;

    GraphInfoWrapper(const GraphInfoWrapper&) = delete;
    GraphInfoWrapper(GraphInfoWrapper&&) = delete;
    auto operator=(const GraphInfoWrapper&) -> GraphInfoWrapper& = delete;
    auto operator=(GraphInfoWrapper&&) -> GraphInfoWrapper& = delete;

    ~GraphInfoWrapper() {
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

    auto loadFromSharedLibrary(const std::filesystem::path& modelPath) {
        m_libModelHandle =
            dlopen(modelPath.string().data(), RTLD_NOW | RTLD_LOCAL);

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

    auto getPtr() -> GraphInfoT*** { return &m_graphsInfo; }

    auto accessGraphs() -> auto& { return m_graphsInfo; }

    auto setGraph() {
        m_graphInfo = std::unique_ptr<GraphInfoT>(m_graphsInfo[0] /* NOLINT */);
    }

    auto getGraphsCountPtr() -> uint32_t* { return &m_graphsCount; }

    auto getGraphCount() const { return m_graphsCount; }

    auto accessGraphCount() -> auto& { return m_graphsCount; }

    auto getGraph() -> auto& { return m_graphInfo->graph; }

    auto accessGraph() -> auto& { return m_graphInfo; }

    auto setComposeGraphsFnHandle(
        ComposeGraphsFnHandleTypeT composeGraphsFnHandle) -> STATUS {
        m_composeGraphsFnHandle = composeGraphsFnHandle;

        if (m_composeGraphsFnHandle == nullptr) {
            return STATUS::FAIL;
        }

        return STATUS::SUCCESS;
    }

    auto setFreeGraphInfoFnHandle(
        FreeGraphInfoFnHandleTypeT freeGraphInfoFnHandle) -> STATUS {
        m_freeGraphInfoFnHandle = freeGraphInfoFnHandle;

        if (m_freeGraphInfoFnHandle == nullptr) {
            return STATUS::FAIL;
        }

        return STATUS::SUCCESS;
    }

    auto composeGraphs(Qnn_BackendHandle_t& qnnBackendHandle,
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

    auto retrieveGraphFromContext(
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

    auto getInputs() -> nonstd::span<Qnn_Tensor_t> {
        return {m_graphInfo->inputTensors, m_graphInfo->numInputTensors};
    }

    auto getOutputs() -> nonstd::span<Qnn_Tensor_t> {
        return {m_graphInfo->outputTensors, m_graphInfo->numOutputTensors};
    }

    auto getNumInputs() const { return m_graphInfo->numInputTensors; }

    auto getNumOutputs() const { return m_graphInfo->numOutputTensors; }

    auto operator[](const size_t index) -> auto& {
        return (*m_graphsInfo)[index] /* NOLINT */;
    }

    static auto copyGraphsInfoV1(
        const QnnSystemContext_GraphInfoV1_t* graphInfoSrc,
        GraphInfoT* graphInfoDst) -> bool {
        graphInfoDst->graphName = nullptr;
        if (graphInfoSrc->graphName != nullptr) {
            graphInfoDst->graphName = strndup(graphInfoSrc->graphName,
                                              strlen(graphInfoSrc->graphName));
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
                == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1)
            {
                if (!copyGraphsInfoV1(&graphsInput[gIdx].graphInfoV1,
                                      &graphInfoArr[gIdx]))
                {
                    return false;
                }
            }
        }

        return true;
    }

    auto copyMetadataToGraphsInfo(
        const QnnSystemContext_BinaryInfo_t* binaryInfo) -> bool {
        if (nullptr == binaryInfo) {
            return false;
        }
        m_graphsCount = 0;
        if (binaryInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
            if (binaryInfo->contextBinaryInfoV1.graphs != nullptr) {
                if (!copyGraphsInfo(binaryInfo->contextBinaryInfoV1.graphs,
                                    binaryInfo->contextBinaryInfoV1.numGraphs))
                {
                    return false;
                }
                m_graphsCount = binaryInfo->contextBinaryInfoV1.numGraphs;
                return true;
            }
        } else if (binaryInfo->version
                   == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2)
        {
            if (binaryInfo->contextBinaryInfoV2.graphs != nullptr) {
                if (!copyGraphsInfo(binaryInfo->contextBinaryInfoV2.graphs,
                                    binaryInfo->contextBinaryInfoV2.numGraphs))
                {
                    return false;
                }
                m_graphsCount = binaryInfo->contextBinaryInfoV2.numGraphs;
                return true;
            }
        }
        return false;
    }

  private:
    std::vector<GraphInfoT> m_graphs;
    std::vector<GraphInfoT*> m_graphPtrs;

    GraphInfoT** m_graphsInfo {};
    uint32_t m_graphsCount {};

    std::unique_ptr<GraphInfoT> m_graphInfo;

    ComposeGraphsFnHandleTypeT m_composeGraphsFnHandle {};
    FreeGraphInfoFnHandleTypeT m_freeGraphInfoFnHandle {};

    void* m_libModelHandle {};
};

}  // namespace edge::qnn
