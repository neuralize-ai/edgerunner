/**
 * @file tensor.hpp
 * @brief Definition of the Tensor class, an opaque handler for model input and
 * output data
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <nonstd/span.hpp>

namespace edge {

/**
 * @brief Enum class representing different types of tensors
 *
 * This enum class defines types of input and output Tensors for a Model.
 * Each type corresponds to a specific data type that the tensor can
 * hold.
 *
 * Possible values:
 * - UNSUPPORTED: Represents an unsupported tensor type
 * - NOTYPE: Represents a tensor with no specific data type
 * - FLOAT32: Represents a tensor with 32-bit floating point data type
 * - FLOAT16: Represents a tensor with 16-bit floating point data type
 * - INT32: Represents a tensor with 32-bit integer data type
 * - UINT32: Represents a tensor with unsigned 32-bit integer data type
 * - INT8: Represents a tensor with 8-bit integer data type
 * - UINT8: Represents a tensor with unsigned 8-bit integer data type
 *
 * @note This enum class is used to specify the data type of a Tensor
 */
enum class TensorType : uint8_t {
    UNSUPPORTED,
    NOTYPE,
    FLOAT32,
    FLOAT16,
    INT32,
    UINT32,
    INT8,
    UINT8,
};

/**
 * @brief A base class for representing a tensor object
 *
 * This class defines the interface for a tensor object, providing methods to
 * retrieve information about the tensor such as its name, type, dimensions,
 * size, and data.
 *
 * @note This class is meant to be used as a base class and should be subclassed
 * to provide concrete implementations.
 */
class Tensor {
  public:
    /**
     * @brief Default constructor for Tensor
     */
    Tensor() = default;

    /**
     * @brief Copy constructor for Tensor
     */
    Tensor(const Tensor&) = default;

    /**
     * @brief Move constructor for Tensor
     */
    Tensor(Tensor&&) = default;

    /**
     * @brief Copy assignment operator for Tensor
     */
    auto operator=(const Tensor&) -> Tensor& = default;

    /**
     * @brief Move assignment operator for Tensor
     */
    auto operator=(Tensor&&) -> Tensor& = default;

    /**
     * @brief Virtual destructor for Tensor
     */
    virtual ~Tensor() = default;

    /**
     * @brief Get the name of the tensor
     *
     * @return The name of the tensor as a string
     */
    virtual auto getName() const -> std::string = 0;

    /**
     * @brief Get the type of the tensor
     *
     * @return The type of the tensor as a TensorType enum
     */
    virtual auto getType() const -> TensorType = 0;

    /**
     * @brief Get the dimensions of the tensor
     *
     * @return A vector of size_t representing the dimensions of the tensor
     */
    virtual auto getDimensions() const -> std::vector<size_t> = 0;

    /**
     * @brief Get the total size of the tensor
     *
     * @return The total size of the tensor as a size_t
     */
    virtual auto getSize() const -> size_t = 0;

    /**
     * @brief Get a non-owning span of the tensor data casted to type T
     *
     * Use getType() to ensure data is accessed correctly
     *
     * @tparam T The type to cast the tensor data to
     * @return A non-owning span of type T representing the tensor data
     */
    template<typename T>
    auto getTensorAs() -> nonstd::span<T>;

  protected:
    /**
     * @brief Get a pointer to the raw data of the tensor
     *
     * @return A void pointer to the raw data of the tensor
     */
    virtual auto getDataPtr() -> void* = 0;

    /**
     * @brief Get the total number of bytes in the tensor data
     *
     * @return The total number of bytes in the tensor data as a size_t
     */
    virtual auto getNumBytes() -> size_t = 0;
};

template<typename T>
auto Tensor::getTensorAs() -> nonstd::span<T> {
    auto* dataPtr = getDataPtr();

    if (dataPtr == nullptr) {
        return {};
    }

    const auto numBytes = getNumBytes();
    const auto numElementBytes = sizeof(T);

    const auto numElements = numBytes / numElementBytes;
    return {static_cast<T*>(dataPtr), numElements};
}

}  // namespace edge
