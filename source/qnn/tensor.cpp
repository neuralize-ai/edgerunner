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

}  // namespace edge::qnn
