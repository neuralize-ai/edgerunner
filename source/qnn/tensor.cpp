#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <string>
#include <variant>
#include <vector>

#include "edgerunner/tensor.hpp"

#include <QnnTypes.h>
#include <nonstd/span.hpp>

#include "edgerunner/qnn/tensor.hpp"
#include "edgerunner/qnn/tensorOps.hpp"

namespace edge::qnn {

TensorImpl::TensorImpl(Qnn_Tensor_t* qnnTensor, const bool allocate)
    : m_tensor(qnnTensor) {
    if (!allocate) {
        return;
    }

    /* TODO: use memhandle */
    setQnnTensorMemType(*m_tensor, QNN_TENSORMEMTYPE_RAW);

    Qnn_ClientBuffer_t clientBuffer = QNN_CLIENT_BUFFER_INIT;

    const auto numBytes = getNumBytes();

    m_data.resize(numBytes);

    clientBuffer.data = m_data.data();
    clientBuffer.dataSize = static_cast<uint32_t>(numBytes);

    setQnnTensorClientBuf(*m_tensor, clientBuffer);
}

auto TensorImpl::getName() const -> std::string {
    if (m_tensor == nullptr) {
        return "";
    }

    auto tensorVariant = getTensorTypeVariant(*m_tensor);
    return std::visit([](auto&& tensor) { return tensor.get().name; },
                      tensorVariant);
}

auto TensorImpl::getType() const -> TensorType {
    if (m_tensor == nullptr) {
        return TensorType::NOTYPE;
    }

    auto tensorVariant = getTensorTypeVariant(*m_tensor);
    const auto qnnDataType = std::visit(
        [](auto&& tensor) { return tensor.get().dataType; }, tensorVariant);

    switch (qnnDataType) {
        case QNN_DATATYPE_FLOAT_16:
            return TensorType::FLOAT16;
        case QNN_DATATYPE_FLOAT_32:
            return TensorType::FLOAT32;
        case QNN_DATATYPE_INT_8:
            return TensorType::INT8;
        case QNN_DATATYPE_INT_16:
            return TensorType::INT16;
        case QNN_DATATYPE_INT_32:
            return TensorType::INT32;
        case QNN_DATATYPE_UINT_8:
            return TensorType::UINT8;
        case QNN_DATATYPE_UINT_16:
            return TensorType::UINT16;
        case QNN_DATATYPE_UINT_32:
            return TensorType::UINT32;
        case QNN_DATATYPE_SFIXED_POINT_8:
            return TensorType::INT8;
        case QNN_DATATYPE_SFIXED_POINT_16:
            return TensorType::INT16;
        case QNN_DATATYPE_SFIXED_POINT_32:
            return TensorType::INT32;
        case QNN_DATATYPE_UFIXED_POINT_8:
            return TensorType::UINT8;
        case QNN_DATATYPE_UFIXED_POINT_16:
            return TensorType::UINT16;
        case QNN_DATATYPE_UFIXED_POINT_32:
            return TensorType::UINT32;
        default:
            return TensorType::UNSUPPORTED;
    }
}

auto TensorImpl::getDimensions() const -> std::vector<size_t> {
    if (m_tensor == nullptr) {
        return {};
    }

    auto tensorVariant = getTensorTypeVariant(*m_tensor);

    const auto qnnDimensions = std::visit(
        [](auto&& tensor) {
            return nonstd::span<uint32_t> {tensor.get().dimensions,
                                           tensor.get().rank};
        },
        tensorVariant);

    return {qnnDimensions.cbegin(), qnnDimensions.cend()};
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

    return getTensorMemoryPtr(*m_tensor);
}

auto TensorImpl::getNumBytes() -> size_t {
    if (m_tensor == nullptr) {
        return {};
    }

    size_t numBytes = 0;

    const auto type = getType();

    switch (type) {
        case TensorType::FLOAT16:
            numBytes = 2;
            break;
        case TensorType::FLOAT32:
            numBytes = sizeof(float);
            break;
        case TensorType::INT8:
            numBytes = sizeof(int8_t);
            break;
        case TensorType::INT16:
            numBytes = sizeof(int16_t);
            break;
        case TensorType::INT32:
            numBytes = sizeof(int32_t);
            break;
        case TensorType::UINT8:
            numBytes = sizeof(uint8_t);
            break;
        case TensorType::UINT16:
            numBytes = sizeof(uint16_t);
            break;
        case TensorType::UINT32:
            numBytes = sizeof(uint32_t);
            break;
        default:
            return {};
    }

    return numBytes * getSize();
}

}  // namespace edge::qnn
