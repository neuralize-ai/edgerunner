/**
 * @file Graphs.hpp
 * @brief Header file for the Graphs class, which manages QNN graphs.
 */

#pragma once

#include <cstring>

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

/**
 * @brief Enum representing possible errors that can occur during graph
 * operations.
 */
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

/**
 * @brief Struct representing information about a graph.
 */
using GraphInfoT = struct GraphInfo {
    Qnn_GraphHandle_t graph;
    char* graphName;
    Qnn_Tensor_t* inputTensors;
    uint32_t numInputTensors;
    Qnn_Tensor_t* outputTensors;
    uint32_t numOutputTensors;
};

/**
 * @brief Struct representing configuration information for a graph.
 */
using GraphConfigInfoT = struct GraphConfigInfo {
    char* graphName;
    const QnnGraph_Config_t** graphConfigs;
};

/**
 * @brief Function pointer type for composing graphs.
 */
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

/**
 * @brief Function pointer type for freeing graph information.
 */
using FreeGraphInfoFnHandleTypeT = GraphErrorT (*)(GraphInfoT***, uint32_t);

/**
 * @brief Class for managing QNN graphs.
 */
class Graph {
  public:
    Graph() = default;

    Graph(const Graph&) = delete;
    Graph(Graph&&) = delete;
    auto operator=(const Graph&) -> Graph& = delete;
    auto operator=(Graph&&) -> Graph& = delete;

    ~Graph();

    /**
     * @brief Get the input tensors for the current graph.
     * @return A span of input tensors.
     */
    auto getInputs() -> nonstd::span<Qnn_Tensor_t>;

    /**
     * @brief Get the output tensors for the current graph.
     * @return A span of output tensors.
     */
    auto getOutputs() -> nonstd::span<Qnn_Tensor_t>;

    /**
     * Loads a model from a shared library located at the specified path.
     *
     * @param modelPath The path to the shared library containing the model.
     * @return STATUS The status of the operation (SUCCESS or ERROR).
     */
    auto loadFromSharedLibrary(const std::filesystem::path& modelPath)
        -> STATUS;

    /**
     * Creates a context for the QNN interface with the specified backend and
     * device handles.
     *
     * Graph keeps a reference to the qnnInterface
     *
     * @param qnnInterface The handle of the QNN interface.
     * @param backendHandle The handle to the QNN backend.
     * @param deviceHandle The handle to the QNN device.
     * @return STATUS The status of the operation (SUCCESS or ERROR).
     */
    auto createContext(QNN_INTERFACE_VER_TYPE& qnnInterface,
                       Qnn_BackendHandle_t& backendHandle,
                       Qnn_DeviceHandle_t& deviceHandle) -> STATUS;

    /**
     * Composes graphs using the specified QNN backend handle.
     *
     * @param qnnBackendHandle The handle to the QNN backend.
     * @return STATUS The status of the operation (SUCCESS or ERROR).
     */
    auto composeGraphs(Qnn_BackendHandle_t& qnnBackendHandle) -> STATUS;

    /**
     * @brief Sets the configuration for the composed graphs.
     * @param delegate The delegate for the operation.
     * @param precision The precision of the operation.
     * @return The status of the operation.
     */
    auto setGraphConfig(DELEGATE delegate, TensorType precision) -> STATUS;

    /**
     * @brief Finalizes the composed graphs.
     * @return The status of the operation.
     */
    auto finalizeGraphs() -> STATUS;

    /**
     * Loads the system library required for loading a cached context from a
     * binary buffer
     *
     * @return STATUS indicating the success or failure of loading the system
     * library.
     */
    auto loadSystemLibrary() -> STATUS;

    /**
     * Loads the context from a binary model buffer.
     *
     * Graph keeps a reference to the qnnInterface
     *
     * @param qnnInterface The handle of the QNN interface.
     * @param backendHandle The handle to the QNN backend.
     * @param deviceHandle The handle to the QNN device.
     * @param modelBuffer The binary model buffer containing the model data.
     * @return STATUS indicating the success or failure of loading the context
     * from the binary model buffer.
     */
    auto loadContextFromBinary(QNN_INTERFACE_VER_TYPE& qnnInterface,
                               Qnn_BackendHandle_t& backendHandle,
                               Qnn_DeviceHandle_t& deviceHandle,
                               const nonstd::span<uint8_t>& modelBuffer)
        -> STATUS;

    /**
     * @brief Saves the current context to a binary file.
     * @param binaryPath The path to save the context binary file.
     * @return The status of the operation.
     */
    auto saveContextBinary(const std::filesystem::path& binaryPath) -> STATUS;

    /**
     * @brief Retrieves a graph from the current context.
     *
     * This function retrieves a graph from the current context and returns a
     * status code indicating the success or failure of the operation.
     *
     * @return STATUS - A status code indicating the success or failure of the
     * operation.
     */
    auto retrieveGraphFromContext() -> STATUS;

    /**
     * @brief Executes the graph.
     *
     * This function executes the graph and returns a status code indicating the
     * success or failure of the operation.
     *
     * @return STATUS - A status code indicating the success or failure of the
     * operation.
     */
    auto execute() -> STATUS;

  private:
    auto setGraph() { m_graphInfo = m_graphsInfo[0] /* NOLINT */; }

    auto setComposeGraphsFnHandle(
        ComposeGraphsFnHandleTypeT composeGraphsFnHandle) -> STATUS;

    auto setFreeGraphInfoFnHandle(
        FreeGraphInfoFnHandleTypeT freeGraphInfoFnHandle) -> STATUS;

    auto copyGraphsInfoV1(const QnnSystemContext_GraphInfoV1_t* graphInfoSrc,
                          GraphInfoT* graphInfoDst) -> bool;

    auto copyGraphsInfo(const QnnSystemContext_GraphInfo_t* graphsInput,
                        uint32_t numGraphs) -> bool;

    auto copyMetadataToGraphsInfo(
        const QnnSystemContext_BinaryInfo_t* binaryInfo) -> bool;

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

    QNN_INTERFACE_VER_TYPE m_qnnInterface {};

    QNN_SYSTEM_INTERFACE_VER_TYPE m_qnnSystemInterface =
        QNN_SYSTEM_INTERFACE_VER_TYPE_INIT;
};

}  // namespace edge::qnn
