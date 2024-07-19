#include <cstddef>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

#include "edgerunner/tensor.hpp"

#include <tensorflow/lite/core/c/c_api_types.h>

#include "edgerunner/tflite/tensor.hpp"

namespace edge::tflite {

auto TensorImpl::getName() const -> std::string {
    if (m_tensor == nullptr) {
        return "";
    }
    return m_tensor->name;
}

auto TensorImpl::getType() const -> TensorType {
    if (m_tensor == nullptr) {
        return TensorType::NOTYPE;
    }

    switch (m_tensor->type) {
        case kTfLiteFloat16:
            return TensorType::FLOAT16;
        case kTfLiteFloat32:
            return TensorType::FLOAT32;
        case kTfLiteInt8:
            return TensorType::INT8;
        case kTfLiteInt16:
            return TensorType::INT16;
        case kTfLiteInt32:
            return TensorType::INT32;
        case kTfLiteUInt8:
            return TensorType::UINT8;
        case kTfLiteUInt16:
            return TensorType::UINT16;
        case kTfLiteUInt32:
            return TensorType::UINT32;

        default:
            return TensorType::UNSUPPORTED;
    }
}

auto TensorImpl::getDimensions() const -> std::vector<size_t> {
    if (m_tensor == nullptr) {
        return {};
    }

    std::vector<size_t> dimensions;
    dimensions.reserve(static_cast<size_t>(m_tensor->dims->size));
    for (int i = 0; i < m_tensor->dims->size; ++i) {
        dimensions.push_back(static_cast<size_t>(m_tensor->dims->data[i]));
    }

    return dimensions;
}

auto TensorImpl::getSize() const -> size_t {
    if (m_tensor == nullptr) {
        return {};
    }

    const auto dimensions = getDimensions();
    return static_cast<size_t>(std::accumulate(
        dimensions.cbegin(), dimensions.cend(), 1, std::multiplies<>()));
}

auto TensorImpl::getDataPtr() -> void* {
    if (m_tensor == nullptr) {
        return nullptr;
    }
    return m_tensor->data.data;
}

auto TensorImpl::getNumBytes() -> size_t {
    if (m_tensor == nullptr) {
        return {};
    }

    return m_tensor->bytes;
}

}  // namespace edge::tflite
