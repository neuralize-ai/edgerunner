#include <cstddef>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

#include "edgerunner/tensor.hpp"

#include <QnnTypes.h>

#include "edgerunner/qnn/tensor.hpp"

namespace edge::qnn {

void setQnnTensorMemType(Qnn_Tensor_t& qnn_tensor,
                         Qnn_TensorMemType_t mem_type) {
    if (QNN_TENSOR_VERSION_1 == qnn_tensor.version) {
        qnn_tensor.v1.memType = mem_type;
        return;
    }

#ifdef QNN_TENSOR_V2_INIT
    if (QNN_TENSOR_VERSION_2 == qnn_tensor.version) {
        qnn_tensor.v2.memType = mem_type;
    }
#endif  // QNN_TENSOR_V2_INIT
}

void setQnnTensorClientBuf(Qnn_Tensor_t& qnn_tensor,
                           Qnn_ClientBuffer_t& client_buf) {
    if (QNN_TENSOR_VERSION_1 == qnn_tensor.version) {
        qnn_tensor.v1.clientBuf = client_buf;
        return;
    }

#ifdef QNN_TENSOR_V2_INIT
    if (QNN_TENSOR_VERSION_2 == qnn_tensor.version) {
        qnn_tensor.v2.clientBuf = client_buf;
    }
#endif  // QNN_TENSOR_V2_INIT
}

TensorImpl::TensorImpl(Qnn_Tensor_t* qnnTensor)
    : m_tensor(qnnTensor) {
    setQnnTensorMemType(*m_tensor, QNN_TENSORMEMTYPE_RAW);
    Qnn_ClientBuffer_t clientBuffer = QNN_CLIENT_BUFFER_INIT;

    const auto numBytes = getNumBytes();

    clientBuffer.data = malloc(numBytes);
    clientBuffer.dataSize = numBytes;

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

    uint32_t qnnRank = 0;
    uint32_t* qnnDimensions = nullptr;
    switch (m_tensor->version) {
        case QNN_TENSOR_VERSION_1:
            qnnRank = m_tensor->v1.rank;
            qnnDimensions = m_tensor->v1.dimensions;
            break;
        case QNN_TENSOR_VERSION_2:
            qnnRank = m_tensor->v2.rank;
            qnnDimensions = m_tensor->v2.dimensions;
            break;
        default:
            return {};
    }

    std::vector<size_t> dimensions;
    dimensions.reserve(static_cast<size_t>(qnnRank));
    for (uint32_t i = 0; i < qnnRank; ++i) {
        dimensions.push_back(static_cast<size_t>(qnnDimensions[i]));
    }

    return dimensions;
}

}  // namespace edge::qnn
