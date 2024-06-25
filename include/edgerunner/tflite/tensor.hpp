/**
 * @file tensor.hpp
 * @brief Definition of the TensorImpl class, a concrete implementation of the
 * Tensor interface.
 */

#pragma once

#include <vector>

#include <edgerunner/edgerunner_export.hpp>
#include <tensorflow/lite/core/c/common.h>

#include "edgerunner/tensor.hpp"

namespace edge::tflite {

/**
 * @class TensorImpl
 * @brief Concrete implementation of the Tensor interface for TensorFlow Lite.
 */
class EDGERUNNER_EXPORT TensorImpl final : public Tensor {
  public:
    /**
     * @brief Constructor for TensorImpl.
     * @param tfLiteTensor Pointer to the TfLiteTensor object.
     */
    explicit TensorImpl(TfLiteTensor* tfLiteTensor = nullptr)
        : m_tensor(tfLiteTensor) {}

    TensorImpl(const TensorImpl& other) = default;
    TensorImpl(TensorImpl&&) = default;
    auto operator=(const TensorImpl&) -> TensorImpl& = default;
    auto operator=(TensorImpl&&) -> TensorImpl& = default;

    ~TensorImpl() final = default;

    /**
     * @brief Get the name of the tensor.
     * @return The name of the tensor as a string.
     */
    auto getName() const -> std::string final;

    /**
     * @brief Get the type of the tensor.
     * @return The type of the tensor as a TensorType enum.
     */
    auto getType() const -> TensorType final;

    /**
     * @brief Get the dimensions of the tensor.
     * @return A vector of size_t representing the dimensions of the tensor.
     */
    auto getDimensions() const -> std::vector<size_t> final;

    /**
     * @brief Get the total size of the tensor.
     * @return The total size of the tensor in number of elements.
     */
    auto getSize() const -> size_t final;

  protected:
    /**
     * @brief Get a pointer to the data of the tensor.
     * @return A void pointer to the data of the tensor.
     */
    auto getDataPtr() -> void* final;

    /**
     * @brief Get the number of bytes occupied by the tensor data.
     * @return The number of bytes occupied by the tensor data.
     */
    auto getNumBytes() -> size_t final;

  private:
    EDGERUNNER_SUPPRESS_C4251
    TfLiteTensor* m_tensor;  ///< The underlying TFlite tensor
};

}  // namespace edge::tflite
