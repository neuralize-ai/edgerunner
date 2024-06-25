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
enum class DELEGATE {
    CPU, /**< CPU delegate */
    GPU, /**< GPU delegate */
    NPU /**< NPU delegate */
};

/**
 * @enum STATUS
 * @brief Enum class representing the status of an operation.
 */
enum class STATUS {
    SUCCESS, /**< Operation was successful */
    FAIL /**< Operation failed */
};

/**
 * @brief Reports the name of the library
 *
 * Please see the note above for considerations when creating shared libraries.
 */
class EDGERUNNER_EXPORT Model {
  public:
    explicit Model(const std::filesystem::path& modelPath)
        : m_name(modelPath.stem().string()) {}

    Model(const Model&) = default;
    Model(Model&&) = delete;
    auto operator=(const Model&) -> Model& = default;
    auto operator=(Model&&) -> Model& = delete;

    virtual ~Model() = default;

    virtual void loadModel(const std::filesystem::path& modelPath) = 0;

    auto getNumInputs() const -> size_t { return m_inputs.size(); }

    auto getNumOutputs() const -> size_t { return m_outputs.size(); }

    auto getInput(size_t index) const -> std::shared_ptr<Tensor> {
        if (index < getNumInputs()) {
            return m_inputs[index];
        }

        return {};
    }

    auto getOutput(size_t index) const -> std::shared_ptr<Tensor> {
        if (index < getNumOutputs()) {
            return m_outputs[index];
        }

        return {};
    }

    auto getDelegate() const -> DELEGATE { return m_delegate; }

    virtual auto applyDelegate(const DELEGATE& delegate) -> STATUS = 0;

    virtual auto execute() -> STATUS = 0;

    /**
     * @brief Returns a non-owning pointer to the string stored in this class
     */
    auto name() const -> char const* { return m_name.c_str(); }

  protected:
    auto accessInputs() -> std::vector<std::shared_ptr<Tensor>>& {
        return m_inputs;
    }

    auto accessOutputs() -> std::vector<std::shared_ptr<Tensor>>& {
        return m_outputs;
    }

    void setDelegate(const DELEGATE& delegate) { m_delegate = delegate; }

    void setName(const std::string& name) { m_name = name; }

  private:
    EDGERUNNER_SUPPRESS_C4251
    std::string m_name;

    EDGERUNNER_SUPPRESS_C4251
    std::vector<std::shared_ptr<Tensor>> m_inputs;

    EDGERUNNER_SUPPRESS_C4251
    std::vector<std::shared_ptr<Tensor>> m_outputs;

    EDGERUNNER_SUPPRESS_C4251
    DELEGATE m_delegate = DELEGATE::CPU;
};

}  // namespace edge
