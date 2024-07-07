/**
 * @file model.hpp
 * @brief Definition of the ModelImpl class, which implements the Model
 * interface for TensorFlow Lite models.
 */

#pragma once

#include <tensorflow/lite/core/c/c_api_types.h>
#include <tensorflow/lite/core/c/common.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>

#include "edgerunner/model.hpp"

namespace edge::tflite {

/**
 * @class ModelImpl
 * @brief Implementation of the Model interface for TensorFlow Lite models.
 */
class ModelImpl final : public Model {
  public:
    /**
     * @brief Constructor for ModelImpl.
     * @param modelPath The path to the TensorFlow Lite model file.
     */
    explicit ModelImpl(const std::filesystem::path& modelPath)
        : Model(modelPath) {
        loadModel(modelPath);
        createInterpreter();
        allocate();
    }

    /**
     * @brief Constructor for ModelImpl.
     * @param modelPath The path to the TensorFlow Lite model file.
     */
    explicit ModelImpl(const nonstd::span<uint8_t>& modelBuffer) {
        loadModel(modelBuffer);
        createInterpreter();
        allocate();
    }

    ModelImpl(const ModelImpl&) = delete;
    ModelImpl(ModelImpl&&) = delete;
    auto operator=(const ModelImpl&) -> ModelImpl& = delete;
    auto operator=(ModelImpl&&) -> ModelImpl& = delete;

    /**
     * @brief Destructor for ModelImpl.
     */
    ~ModelImpl() final;

    /**
     * @brief Loads the TensorFlow Lite model from the specified path.
     *
     * This function loads a TensorFlow Lite model from the specified file path.
     * The model file should be in the TensorFlow Lite format.
     *
     * @param modelPath The path to the TensorFlow Lite model file.
     * @return STATUS Returns a status indicating whether the model was
     * successfully loaded or not.
     */
    auto loadModel(const std::filesystem::path& modelPath) -> STATUS final;

    /**
     * @brief Loads the TensorFlow Lite model from the specified buffer.
     *
     * This function loads a TensorFlow Lite model from the provided buffer. The
     * buffer should contain the raw data of the TensorFlow Lite model.
     *
     * @param modelBuffer The buffer containing the TensorFlow Lite model.
     * @return STATUS Returns a status indicating whether the model was
     * successfully loaded or not.
     */
    auto loadModel(const nonstd::span<uint8_t>& modelBuffer) -> STATUS final;

    /**
     * @brief Applies a delegate to the TensorFlow Lite interpreter.
     * @param delegate The delegate to apply.
     * @return The status of the operation.
     */
    auto applyDelegate(const DELEGATE& delegate) -> STATUS final;

    /**
     * @brief Executes the TensorFlow Lite model.
     * @return The status of the operation.
     */
    auto execute() -> STATUS final;

  private:
    /**
     * Creates a new interpreter object.
     *
     * This function initializes a new interpreter object and sets up any
     * necessary resources.
     */
    void createInterpreter();

    /**
     * Allocates memory for the interpreter.
     *
     * This function allocates memory for the interpreter, including input and
     * output tensors
     */
    void allocate();

    /**
     * Deletes the delegate object.
     *
     * This function deletes the delegate object and frees up any resources it
     * was using.
     */
    void deleteDelegate();

    std::filesystem::path
        m_modelPath;  ///< The path to the TensorFlow Lite model file

    std::unique_ptr<::tflite::FlatBufferModel>
        m_modelBuffer;  ///< The TensorFlow Lite model buffer

    std::unique_ptr<::tflite::Interpreter>
        m_interpreter;  ///< The TensorFlow Lite interpreter

    TfLiteDelegate* m_delegate = nullptr;  ///< The TensorFlow Lite delegate
};

}  // namespace edge::tflite
