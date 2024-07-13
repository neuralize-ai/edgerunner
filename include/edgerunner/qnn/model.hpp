/**
 * @file model.hpp
 * @brief Definition of the ModelImpl class, which implements the Model
 * interface for QNN models.
 */

#pragma once

#include <QnnInterface.h>
#include <System/QnnSystemInterface.h>
#include <dlfcn.h>

#include "edgerunner/model.hpp"
#include "edgerunner/qnn/backend.h"

namespace edge::qnn {

using ModelErrorT = enum ModelError {
    MODEL_NO_ERROR = 0,
    MODEL_TENSOR_ERROR = 1,
    MODEL_PARAMS_ERROR = 2,
    MODEL_NODES_ERROR = 3,
    MODEL_GRAPH_ERROR = 4,
    MODEL_CONTEXT_ERROR = 5,
    MODEL_GENERATION_ERROR = 6,
    MODEL_SETUP_ERROR = 7,
    MODEL_INVALID_ARGUMENT_ERROR = 8,
    MODEL_FILE_ERROR = 9,
    MODEL_MEMORY_ALLOCATE_ERROR = 10,
    // Value selected to ensure 32 bits.
    MODEL_UNKNOWN_ERROR = 0x7FFFFFFF
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
    ModelErrorT (*)(Qnn_BackendHandle_t,
                    QnnInterface_ImplementationV2_16_t,
                    Qnn_ContextHandle_t,
                    const GraphConfigInfoT**,
                    const uint32_t,
                    GraphInfoT***,
                    uint32_t*,
                    bool,
                    QnnLog_Callback_t,
                    QnnLog_Level_t);

using FreeGraphInfoFnHandleTypeT = ModelErrorT (*)(GraphInfoT***, uint32_t);

/**
 * @class ModelImpl
 * @brief Implementation of the Model interface for QNN models.
 */
class ModelImpl final : public Model {
  public:
    /**
     * @brief Constructor for ModelImpl.
     * @param modelPath The path to the QNN model file.
     */
    explicit ModelImpl(const std::filesystem::path& modelPath)
        : Model(modelPath) {
    /**
     * @brief Constructor for ModelImpl.
     * @param modelPath The path to the QNN model file.
     */
    explicit ModelImpl(const nonstd::span<uint8_t>& modelBuffer) {
        loadModel(modelBuffer);
    }

    ModelImpl(const ModelImpl&) = delete;
    ModelImpl(ModelImpl&&) = delete;
    auto operator=(const ModelImpl&) -> ModelImpl& = delete;
    auto operator=(ModelImpl&&) -> ModelImpl& = delete;

    /**
     * @brief Loads the QNN model from the specified path.
     *
     * This function loads a QNN model from the specified file path.
     * The model file should be in the QNN context binary format.
     *
     * @param modelPath The path to the QNN model file.
     * @return STATUS Returns a status indicating whether the model was
     * successfully loaded or not.
     */
    auto loadModel(const std::filesystem::path& modelPath) -> STATUS final;

    /**
     * @brief Loads the QNN model from the specified buffer.
     *
     * This function loads a QNN model from the provided buffer. The
     * buffer should contain the raw data of the QNN model.
     *
     * @param modelBuffer The buffer containing the QNN model.
     * @return STATUS Returns a status indicating whether the model was
     * successfully loaded or not.
     */
    auto loadModel(const nonstd::span<uint8_t>& modelBuffer) -> STATUS final;

    /**
     * @brief Applies a delegate to the QNN backend.
     * @param delegate The delegate to apply.
     * @return The status of the operation.
     */
    auto applyDelegate(const DELEGATE& delegate) -> STATUS final;

    /**
     * @brief Executes the QNN model.
     * @return The status of the operation.
     */
    auto execute() -> STATUS final;

  private:
    auto loadFromSharedLibrary(const std::filesystem::path& modelPath)
        -> STATUS;

    auto composeGraphs() -> STATUS;

    auto finalizeGraphs() -> STATUS;

    std::filesystem::path m_modelPath;  ///< The path to the QNN model file

    std::unique_ptr<Backend> m_backend;

    /* Graph */
    Qnn_GraphHandle_t m_graph {};

    GraphInfoT** m_graphInfo {};
    uint32_t m_graphsCount {};
    GraphConfigInfoT* m_graphConfigInfo {};

    ComposeGraphsFnHandleTypeT m_composeGraphsFnHandle {};
    FreeGraphInfoFnHandleTypeT m_freeGraphInfoFnHandle {};

    void* m_libModelHandle {};

};

}  // namespace edge::qnn
