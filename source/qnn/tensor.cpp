#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

#include "edgerunner/tensor.hpp"

#include <QnnTypes.h>
#include <nonstd/span.hpp>

#include "edgerunner/qnn/tensor.hpp"

namespace edge::qnn {

void setQnnTensorMemType(Qnn_Tensor_t& qnnTensor, Qnn_TensorMemType_t memType) {
    if (QNN_TENSOR_VERSION_1 == qnnTensor.version) {
        qnnTensor.v1.memType = memType;
        return;
    }

#ifdef QNN_TENSOR_V2_INIT
    if (QNN_TENSOR_VERSION_2 == qnnTensor.version) {
        qnnTensor.v2.memType = memType;
    }
#endif
}

void setQnnTensorClientBuf(Qnn_Tensor_t& qnnTensor,
                           Qnn_ClientBuffer_t& clientBuf) {
    if (QNN_TENSOR_VERSION_1 == qnnTensor.version) {
        qnnTensor.v1.clientBuf = clientBuf;
        return;
    }

#ifdef QNN_TENSOR_V2_INIT
    if (QNN_TENSOR_VERSION_2 == qnnTensor.version) {
        qnnTensor.v2.clientBuf = clientBuf;
    }
#endif
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

    switch (m_tensor->version) {
        case QNN_TENSOR_VERSION_1:
            return m_tensor->v1.name;
        case QNN_TENSOR_VERSION_2:
            return m_tensor->v2.name;
        default:
            return "";
    }
}

auto TensorImpl::getType() const -> TensorType {
    if (m_tensor == nullptr) {
        return TensorType::NOTYPE;
    }

    Qnn_DataType_t qnnDataType {};
    switch (m_tensor->version) {
        case QNN_TENSOR_VERSION_1:
            qnnDataType = m_tensor->v1.dataType;
            break;
        case QNN_TENSOR_VERSION_2:
            qnnDataType = m_tensor->v2.dataType;
            break;
        default:
            return TensorType::NOTYPE;
    }

    switch (qnnDataType) {
        case QNN_DATATYPE_FLOAT_32:
            return TensorType::FLOAT32;
        case QNN_DATATYPE_FLOAT_16:
            return TensorType::FLOAT16;
        case QNN_DATATYPE_INT_32:
            return TensorType::INT32;
        case QNN_DATATYPE_UINT_32:
            return TensorType::UINT32;
        case QNN_DATATYPE_INT_8:
            return TensorType::INT8;
        case QNN_DATATYPE_UINT_8:
            return TensorType::UINT8;
        default:
            return TensorType::UNSUPPORTED;
    }
}

auto TensorImpl::getDimensions() const -> std::vector<size_t> {
    if (m_tensor == nullptr) {
        return {};
    }

    nonstd::span<uint32_t> qnnDimensions;

    switch (m_tensor->version) {
        case QNN_TENSOR_VERSION_1:
            qnnDimensions = nonstd::span<uint32_t> {m_tensor->v1.dimensions,
                                                    m_tensor->v1.rank};
            break;
        case QNN_TENSOR_VERSION_2:
            qnnDimensions = nonstd::span<uint32_t> {m_tensor->v2.dimensions,
                                                    m_tensor->v2.rank};
            break;
        default:
            return {};
    }

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

    switch (m_tensor->version) {
        case QNN_TENSOR_VERSION_1: {
            auto tensor = m_tensor->v1;
            switch (tensor.memType) {
                case QNN_TENSORMEMTYPE_RAW:
                    return tensor.memHandle;
                case QNN_TENSORMEMTYPE_MEMHANDLE:
                    return tensor.clientBuf.data;
                default:
                    return nullptr;
            }
        }
        case QNN_TENSOR_VERSION_2: {
            auto tensor = m_tensor->v2;
            switch (tensor.memType) {
                case QNN_TENSORMEMTYPE_RAW:
                    return tensor.memHandle;
                case QNN_TENSORMEMTYPE_MEMHANDLE:
                    return tensor.clientBuf.data;
                default:
                    return nullptr;
            }
        }
        default:
            return nullptr;
    }
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