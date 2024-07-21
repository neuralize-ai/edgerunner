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

using ContextBinaryInfoVariant =
    std::variant<std::reference_wrapper<QnnSystemContext_BinaryInfoV1_t>,
                 std::reference_wrapper<QnnSystemContext_BinaryInfoV2_t>>;

using ConstContextBinaryInfoVariant =
    std::variant<std::reference_wrapper<const QnnSystemContext_BinaryInfoV1_t>,
                 std::reference_wrapper<const QnnSystemContext_BinaryInfoV2_t>>;

inline auto getContextBinaryInfoVariant(
    QnnSystemContext_BinaryInfo_t* contextBinaryInfo)
    -> ContextBinaryInfoVariant {
    switch (contextBinaryInfo->version) {
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1:
            return std::ref(
                contextBinaryInfo->contextBinaryInfoV1 /* NOLINT */);
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2:
            return std::ref(
                contextBinaryInfo->contextBinaryInfoV2 /* NOLINT */);
        default:
            return std::ref(
                contextBinaryInfo->contextBinaryInfoV1 /* NOLINT */);
    }
}

inline auto getContextBinaryInfoVariant(
    const QnnSystemContext_BinaryInfo_t* contextBinaryInfo)
    -> ConstContextBinaryInfoVariant {
    switch (contextBinaryInfo->version) {
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1:
            return std::cref(
                contextBinaryInfo->contextBinaryInfoV1 /* NOLINT */);
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2:
            return std::cref(
                contextBinaryInfo->contextBinaryInfoV2 /* NOLINT */);
        default:
            return std::cref(
                contextBinaryInfo->contextBinaryInfoV1 /* NOLINT */);
    }
}

class GraphsInfo {
  public:
    GraphsInfo() = default;

    GraphsInfo(const GraphsInfo&) = delete;
    GraphsInfo(GraphsInfo&&) = delete;
    auto operator=(const GraphsInfo&) -> GraphsInfo& = delete;
    auto operator=(GraphsInfo&&) -> GraphsInfo& = delete;

    ~GraphsInfo();

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

    auto loadFromSharedLibrary(const std::filesystem::path& modelPath)
        -> STATUS;

    auto setComposeGraphsFnHandle(
        ComposeGraphsFnHandleTypeT composeGraphsFnHandle) -> STATUS;

    auto setFreeGraphInfoFnHandle(
        FreeGraphInfoFnHandleTypeT freeGraphInfoFnHandle) -> STATUS;

    auto composeGraphs(Qnn_BackendHandle_t& qnnBackendHandle,
                       QnnInterface_ImplementationV2_16_t& qnnInterface,
                       Qnn_ContextHandle_t& qnnContext) -> GraphErrorT;

    auto retrieveGraphFromContext(
        QnnInterface_ImplementationV2_16_t& qnnInterface,
        Qnn_ContextHandle_t& qnnContext) -> STATUS;

    static auto copyGraphsInfoV1(
        const QnnSystemContext_GraphInfoV1_t* graphInfoSrc,
        GraphInfoT* graphInfoDst) -> bool;

    auto copyGraphsInfo(const QnnSystemContext_GraphInfo_t* graphsInput,
                        uint32_t numGraphs) -> bool;

    auto copyMetadataToGraphsInfo(
        const QnnSystemContext_BinaryInfo_t* binaryInfo) -> bool;

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
