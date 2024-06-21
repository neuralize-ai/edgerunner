#pragma once

#include <string>
#include <vector>

#include <nonstd/span.hpp>

#include "edgerunner/edgerunner_export.hpp"

namespace edge {

enum class EDGERUNNER_EXPORT TensorType {
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
