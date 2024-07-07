/**
 * @file model.hpp
 * @brief Definition of the Model class, a base class for machine learning
 * models.
 */

#pragma once

#include <filesystem>
#include <string>

#include <nonstd/span.hpp>

#include "edgerunner/edgerunner_export.hpp"
#include "tensor.hpp"

namespace edge {

/**
 * @enum DELEGATE
 * @brief Enum class representing different types of delegates for model
 * execution.
 */
enum class DELEGATE : uint8_t {
    CPU, /**< CPU delegate */
    GPU, /**< GPU delegate */
    NPU /**< NPU delegate */
};

/**
 * @enum STATUS
 * @brief Enum class representing the status of an operation.
 */
enum class STATUS : uint8_t {
    SUCCESS, /**< Operation was successful */
    FAIL /**< Operation failed */
};

/**
 * @class Model
 * @brief A base class for machine learning models.
 *
 * This class serves as a base class for machine learning models. It provides
 * common functionality such as loading a model, accessing inputs and outputs,
 * applying a delegate, and executing the model.
 */
class EDGERUNNER_EXPORT Model {
  public:
    /**
     * @brief Constructor for the Model class.
     *
     * This constructor initializes a Model object with the given model path.
     *
     * @param modelPath The path to the model file.
     */
    explicit Model(const std::filesystem::path& modelPath)
        : m_name(modelPath.stem().string()) {}

    Model() = default;
    Model(const Model&) = default;
    Model(Model&&) = delete;
    auto operator=(const Model&) -> Model& = default;
    auto operator=(Model&&) -> Model& = delete;

    /**
     * @brief Virtual destructor for the Model class.
     */
    virtual ~Model() = default;

    /**
     * @brief Pure virtual function to load a model from a file path.
     *
     * This function is a pure virtual function that must be implemented by any
     * derived classes. It is used to load a model from a file path.
     *
     * @param modelPath The path to the model file
     * @return STATUS The status of the model loading operation
     */
    virtual auto loadModel(const std::filesystem::path& modelPath)
        -> STATUS = 0;

    /**
     * @brief Pure virtual function to load a model from a file buffer.
     *
     * This function is a pure virtual function that must be implemented by any
     * derived classes. It is used to load a model from a file buffer.
     *
     * @param modelBuffer The buffer containing the model
     * @return STATUS The status of the model loading operation
     */
    virtual auto loadModel(const nonstd::span<uint8_t>& modelBuffer)
        -> STATUS = 0;

    /**
     * @brief Get the number of input tensors in the model.
     *
     * @return The number of input tensors
     */
    auto getNumInputs() const -> size_t { return m_inputs.size(); }

    /**
     * @brief Get the number of output tensors in the model.
     *
     * @return The number of output tensors
     */
    auto getNumOutputs() const -> size_t { return m_outputs.size(); }

    /**
     * @brief Get the input tensor at the specified index.
     *
     * @param index The index of the input tensor
     * @return The input tensor at the specified index, or nullptr if index is
     * out of bounds
     */
    auto getInput(size_t index) const -> std::shared_ptr<Tensor>;

    /**
     * @brief Get the output tensor at the specified index.
     *
     * @param index The index of the output tensor
     * @return The output tensor at the specified index, or nullptr if index is
     * out of bounds
     */
    auto getOutput(size_t index) const -> std::shared_ptr<Tensor>;

    /**
     * @brief Get the inputs of the model.
     *
     * This function returns a reference to a vector of shared pointers to
     * Tensor objects, which represent the inputs of the model.
     *
     * @return A reference to a vector of shared pointers to Tensor objects
     * representing the inputs of the model.
     */
    auto getInputs() -> std::vector<std::shared_ptr<Tensor>>& {
        return m_inputs;
    }

    /**
     * @brief Get the outputs of the model.
     *
     * This function returns a reference to a vector of shared pointers to
     * Tensor objects, which represent the outputs of the model.
     *
     * @return A reference to a vector of shared pointers to Tensor objects
     * representing the outputs of the model.
     */
    auto getOutputs() -> std::vector<std::shared_ptr<Tensor>>& {
        return m_outputs;
    }

    /**
     * @brief Get the current delegate used for model execution.
     *
     * @return The delegate currently set for model execution
     */
    auto getDelegate() const -> DELEGATE { return m_delegate; }

    /**
     * @brief Apply a delegate for model execution.
     *
     * @param delegate The delegate to apply
     * @return The status of the operation
     */
    virtual auto applyDelegate(const DELEGATE& delegate) -> STATUS = 0;

    /**
     * @brief Execute the model.
     *
     * @return The status of the operation
     */
    virtual auto execute() -> STATUS = 0;

    /**
     * @brief Get the name of the model.
     *
     * @return The name of the model
     */
    auto name() const -> const std::string& { return m_name; }

  protected:
    /**
     * @brief Set the delegate for model execution.
     *
     * @param delegate The delegate to set
     */
    void setDelegate(const DELEGATE& delegate) { m_delegate = delegate; }

  private:
    EDGERUNNER_SUPPRESS_C4251
    std::string m_name; /**< Name of the model */

    EDGERUNNER_SUPPRESS_C4251
    std::vector<std::shared_ptr<Tensor>>
        m_inputs; /**< Input tensors of the model */

    EDGERUNNER_SUPPRESS_C4251
    std::vector<std::shared_ptr<Tensor>>
        m_outputs; /**< Output tensors of the model */

    EDGERUNNER_SUPPRESS_C4251
    DELEGATE m_delegate =
        DELEGATE::CPU; /**< Delegate used for model execution */
};

inline auto Model::getInput(size_t index) const -> std::shared_ptr<Tensor> {
    if (index < getNumInputs()) {
        return m_inputs[index];
    }

    return nullptr;
}

inline auto Model::getOutput(size_t index) const -> std::shared_ptr<Tensor> {
    if (index < getNumOutputs()) {
        return m_outputs[index];
    }

    return nullptr;
}

}  // namespace edge
