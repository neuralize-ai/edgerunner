#pragma once

#include <cstring>
#include <memory>

#include <QnnCommon.h>
#include <QnnGraph.h>
#include <QnnInterface.h>
#include <QnnTypes.h>
#include <System/QnnSystemContext.h>
#include <System/QnnSystemInterface.h>
#include <dlfcn.h>
#include <nonstd/span.hpp>

#include "edgerunner/model.hpp"

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

using ComposeGraphsFnHandleTypeT = GraphErrorT (*)(Qnn_BackendHandle_t,
                                                   QNN_INTERFACE_VER_TYPE,
                                                   Qnn_ContextHandle_t,
                                                   const GraphConfigInfoT**,
                                                   const uint32_t,
                                                   GraphInfoT***,
                                                   uint32_t*,
                                                   bool,
                                                   QnnLog_Callback_t,
                                                   QnnLog_Level_t);

using FreeGraphInfoFnHandleTypeT = GraphErrorT (*)(GraphInfoT***, uint32_t);

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

    auto setGraph() { m_graphInfo = m_graphsInfo[0] /* NOLINT */; }

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

    auto setComposeGraphsFnHandle(
        ComposeGraphsFnHandleTypeT composeGraphsFnHandle) -> STATUS;

    auto setFreeGraphInfoFnHandle(
        FreeGraphInfoFnHandleTypeT freeGraphInfoFnHandle) -> STATUS;

    /* shared library setup */
    auto loadFromSharedLibrary(const std::filesystem::path& modelPath)
        -> STATUS;

    auto createContext(QNN_INTERFACE_VER_TYPE& qnnInterface,
                       Qnn_BackendHandle_t& backendHandle,
                       Qnn_DeviceHandle_t& deviceHandle) -> STATUS;

    auto composeGraphs(Qnn_BackendHandle_t& qnnBackendHandle) -> STATUS;

    /* context binary setup */
    auto loadSystemLibrary() -> STATUS;

    auto loadContextFromBinary(QNN_INTERFACE_VER_TYPE& qnnInterface,
                               Qnn_BackendHandle_t& backendHandle,
                               Qnn_DeviceHandle_t& deviceHandle,
                               const nonstd::span<uint8_t>& modelBuffer)
        -> STATUS;

    auto retrieveGraphFromContext() -> STATUS;

    auto copyGraphsInfoV1(const QnnSystemContext_GraphInfoV1_t* graphInfoSrc,
                          GraphInfoT* graphInfoDst) -> bool;

    auto copyGraphsInfo(const QnnSystemContext_GraphInfo_t* graphsInput,
                        uint32_t numGraphs) -> bool;

    auto copyMetadataToGraphsInfo(
        const QnnSystemContext_BinaryInfo_t* binaryInfo) -> bool;

    auto getContext() -> auto& { return m_context; }

    auto getSystemInterface() -> auto& { return m_qnnSystemInterface; }

  private:
    std::vector<GraphInfoT> m_graphs;
    std::vector<GraphInfoT*> m_graphPtrs;

    GraphInfoT* m_graphInfo {};

    GraphInfoT** m_graphsInfo {};
    uint32_t m_graphsCount {};

    ComposeGraphsFnHandleTypeT m_composeGraphsFnHandle {};
    FreeGraphInfoFnHandleTypeT m_freeGraphInfoFnHandle {};

    void* m_libModelHandle {};

    std::vector<Qnn_Tensor_t> m_inputTensors;
    std::vector<Qnn_Tensor_t> m_outputTensors;

    Qnn_ContextHandle_t m_context {};

    QNN_INTERFACE_VER_TYPE m_qnnInterface;

    QNN_SYSTEM_INTERFACE_VER_TYPE m_qnnSystemInterface =
        QNN_SYSTEM_INTERFACE_VER_TYPE_INIT;
};

}  // namespace edge::qnn
