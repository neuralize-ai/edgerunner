#pragma once

#include <cstring>

#include <QnnCommon.h>
#include <QnnGraph.h>
#include <QnnInterface.h>
#include <QnnTypes.h>
#include <System/QnnSystemContext.h>

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

inline auto copyGraphsInfoV1(const QnnSystemContext_GraphInfoV1_t* graphInfoSrc,
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

inline auto copyGraphsInfo(const QnnSystemContext_GraphInfo_t* graphsInput,
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
        free(graphInfoArr);
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

inline auto copyMetadataToGraphsInfo(
    const QnnSystemContext_BinaryInfo_t* binaryInfo,
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

}  // namespace edge::qnn
