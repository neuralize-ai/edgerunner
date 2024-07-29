/**
 * @file model.hpp
 * @brief Definition of the ModelImpl class, which implements the Model
 * interface for QNN models.
 */

#pragma once

#include <QnnInterface.h>
#include <System/QnnSystemInterface.h>
#include <dlfcn.h>

#include "backend.hpp"
#include "edgerunner/model.hpp"
#include "graph.hpp"

namespace edge::qnn {

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
    explicit ModelImpl(const std::filesystem::path& modelPath);

    /**
     * @brief Constructor for ModelImpl.
     * @param modelPath The path to the QNN model file.
     */
    explicit ModelImpl(const nonstd::span<uint8_t>& modelBuffer);

    ModelImpl(const ModelImpl&) = delete;
    ModelImpl(ModelImpl&&) = delete;
    auto operator=(const ModelImpl&) -> ModelImpl& = delete;
    auto operator=(ModelImpl&&) -> ModelImpl& = delete;

    ~ModelImpl() final = default;

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
    /**
     * @brief Loads a QNN model from a shared library.
     *
     * This function loads a QNN model from a shared library located at the
     * specified path.
     *
     * @param modelPath The path to the shared library containing the QNN model.
     * @return STATUS The status of the operation (SUCCESS or FAIL).
     */
    auto loadFromSharedLibrary(const std::filesystem::path& modelPath)
        -> STATUS;
    /**
     * Loads a QNN model from a serialized binary buffer.
     *
     * This function takes a nonstd::span<uint8_t> modelBuffer as input and
     * attempts to load a model from the binary data contained within it.
     *
     * @param modelBuffer A nonstd::span<uint8_t> containing the binary data of
     * the model to be loaded.
     *
     * @return STATUS The status of the operation (SUCCESS or FAIL).
     */
    auto loadFromContextBinary(const nonstd::span<uint8_t>& modelBuffer)
        -> STATUS;

    /**
     * @brief Composes the graphs for the loaded QNN model.
     *
     * This function composes the graphs for the loaded QNN model based on the
     * model configuration.
     *
     * @return STATUS The status of the operation (SUCCESS or FAIL).
     */
    auto composeGraphs() -> STATUS;

    /**
     * Detects graph operation precision
     *
     * This function queries the graph to detect what precision the graph should
     * be executed in. This is required in particular for QNN delegate
     */
    auto detectPrecision() -> TensorType;

    /**
     * @brief Sets the configuration for the composed graphs.
     *
     * This function sets the configuration for the composed graphs, operation
     * precision, graph optimization level.
     *
     * @return STATUS The status of the operation (SUCCESS or FAIL).
     */
    auto setGraphConfig() -> STATUS;

    /**
     * @brief Finalizes the composed graphs.
     *
     * This function finalizes the composed graphs and prepares them for
     * execution.
     *
     * @return STATUS The status of the operation (SUCCESS or FAIL).
     */
    auto finalizeGraphs() -> STATUS;

    /**
     * Saves the current context to a binary file.
     *
     * This function saves the current context to a binary file specified by the
     * input binaryPath.
     *
     * @param binaryPath The path to the binary file where the context will be
     * saved.
     * @return STATUS Returns a STATUS enum indicating the success or failure of
     * the operation.
     */
    auto saveContextBinary(const std::filesystem::path& binaryPath) -> STATUS;

    /**
     * @brief Allocates input and output tensors
     *
     * This function allocates input and output tensors. Should be used before
     * executing.
     *
     * @return STATUS The status of the operation (SUCCESS or FAIL).
     */
    auto allocate() -> STATUS;

    static void initializeBackend() {
        if (m_backend == nullptr) {
            m_backend = std::make_unique<Backend>(DELEGATE::NPU);
        }
    }

    std::filesystem::path m_modelPath;  ///< The path to the QNN model file

    static std::unique_ptr<Backend> m_backend;

    GraphsInfo m_graphInfo;

    bool m_loadCachedBinary {};
};

}  // namespace edge::qnn
