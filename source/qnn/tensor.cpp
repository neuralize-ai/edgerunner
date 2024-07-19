#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <string>
#include <variant>
#include <vector>

#include "edgerunner/tensor.hpp"

#include <QnnCommon.h>
#include <QnnTypes.h>
#include <nonstd/span.hpp>

#include "edgerunner/qnn/tensor.hpp"

namespace edge::qnn {

using TensorVariant = std::variant<std::reference_wrapper<Qnn_TensorV1_t>,
                                   std::reference_wrapper<Qnn_TensorV2_t>>;

auto getTensorTypeVariant(Qnn_Tensor_t& tensor) -> TensorVariant {
    switch (tensor.version) {
#ifdef QNN_TENSOR_V2_INIT
        case QNN_TENSOR_VERSION_2:
            return std::ref(tensor.v2 /* NOLINT */);
#endif
        default:
            return std::ref(tensor.v1 /* NOLINT */);
    }
}

using TensorMemoryVariant =
    std::variant<Qnn_MemHandle_t, std::reference_wrapper<Qnn_ClientBuffer_t>>;

auto getTensorMemoryPtr(Qnn_Tensor_t& tensor) -> void* {
    auto tensorVariant = getTensorTypeVariant(tensor);
    return std::visit(
        [](auto&& typedTensor) {
            switch (typedTensor.get().memType) {
                case QNN_TENSORMEMTYPE_MEMHANDLE:
                    return typedTensor.get().memHandle;  // NOLINT
                case QNN_TENSORMEMTYPE_RAW:
                    return typedTensor.get().clientBuf.data;  // NOLINT
                default:
                    return static_cast<void*>(nullptr);
            }
        },
        tensorVariant);
}

void setQnnTensorMemType(Qnn_Tensor_t& qnnTensor, Qnn_TensorMemType_t memType) {
    auto tensorVariant = getTensorTypeVariant(qnnTensor);
    std::visit([memType](auto&& tensor) { tensor.get().memType = memType; },
               tensorVariant);
}

void setQnnTensorClientBuf(Qnn_Tensor_t& qnnTensor,
                           Qnn_ClientBuffer_t& clientBuf) {
    auto tensorVariant = getTensorTypeVariant(qnnTensor);
    std::visit(
        [&clientBuf](auto&& tensor) {
            tensor.get().clientBuf /* NOLINT */ = clientBuf;
        },
        tensorVariant);
}

TensorImpl::TensorImpl(Qnn_Tensor_t* qnnTensor)
    : m_tensor(qnnTensor) {
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
        case TensorType::FLOAT32:
            numBytes = sizeof(float);
            break;
        case TensorType::FLOAT16:
            numBytes = 2;
            break;
        case TensorType::INT32:
            numBytes = sizeof(int32_t);
            break;
        case TensorType::UINT32:
            numBytes = sizeof(uint32_t);
            break;
        case TensorType::INT8:
            numBytes = sizeof(int8_t);
            break;
        case TensorType::UINT8:
            numBytes = sizeof(uint8_t);
            break;
        default:
            return {};
    }

    return numBytes * getSize();
}

}  // namespace edge::qnn
