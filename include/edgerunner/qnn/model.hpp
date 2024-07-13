/**
 * @file model.hpp
 * @brief Definition of the ModelImpl class, which implements the Model
 * interface for QNN models.
 */

#pragma once

#include "edgerunner/model.hpp"
#include "edgerunner/qnn/backend.h"

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

};

}  // namespace edge::qnn
