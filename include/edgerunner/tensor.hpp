#pragma once

#include <string>
#include <vector>

#include <nonstd/span.hpp>

#include "edgerunner/edgerunner_export.hpp"

namespace edge {

/**
 * @brief Enum class representing different types of tensors
 *
 * This enum class defines different types of tensors that can be used in a
 * system. Each type corresponds to a specific data type that the tensor can
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
 * @note This enum class is used to specify the data type of tensors in the
 * system
 */
enum class TensorType {
    UNSUPPORTED,
    NOTYPE,
    FLOAT32,
    FLOAT16,
    INT32,
    UINT32,
    INT8,
    UINT8,
};

class EDGERUNNER_EXPORT Tensor {
  public:
    Tensor() = default;

    Tensor(const Tensor&) = default;
    Tensor(Tensor&&) = default;
    auto operator=(const Tensor&) -> Tensor& = default;
    auto operator=(Tensor&&) -> Tensor& = default;

    virtual ~Tensor() = default;

    virtual auto getName() const -> std::string = 0;

    virtual auto getType() const -> TensorType = 0;

    virtual auto getDimensions() const -> std::vector<size_t> = 0;

    virtual auto getSize() const -> size_t = 0;

    template<typename T>
    auto getTensorAs() -> nonstd::span<T> {
        auto* dataPtr = getDataPtr();

        if (dataPtr == nullptr) {
            return {};
        }

        const auto numBytes = getNumBytes();
        const auto numElementBytes = sizeof(T);

        const auto numElements = numBytes / numElementBytes;
        return {static_cast<T*>(dataPtr), numElements};
    }

  protected:
    virtual auto getDataPtr() -> void* = 0;

    virtual auto getNumBytes() -> size_t = 0;
};

}  // namespace edge
