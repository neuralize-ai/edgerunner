#pragma once

#include <cstdlib>
#include <functional>
#include <variant>

#include <QnnTypes.h>

namespace edge::qnn {

using TensorVariant = std::variant<std::reference_wrapper<Qnn_TensorV1_t>,
                                   std::reference_wrapper<Qnn_TensorV2_t>>;

using ConstTensorVariant =
    std::variant<std::reference_wrapper<const Qnn_TensorV1_t>,
                 std::reference_wrapper<const Qnn_TensorV2_t>>;

inline auto getTensorTypeVariant(Qnn_Tensor_t& tensor) -> TensorVariant {
    switch (tensor.version) {
#ifdef QNN_TENSOR_V2_INIT
        case QNN_TENSOR_VERSION_2:
            return std::ref(tensor.v2 /* NOLINT */);
#endif
        default:
            return std::ref(tensor.v1 /* NOLINT */);
    }
}

inline auto getTensorTypeVariant(const Qnn_Tensor_t& tensor)
    -> ConstTensorVariant {
    switch (tensor.version) {
#ifdef QNN_TENSOR_V2_INIT
        case QNN_TENSOR_VERSION_2:
            return std::cref(tensor.v2 /* NOLINT */);
#endif
        default:
            return std::cref(tensor.v1 /* NOLINT */);
    }
}

using TensorMemoryVariant =
    std::variant<Qnn_MemHandle_t, std::reference_wrapper<Qnn_ClientBuffer_t>>;

inline auto getTensorMemoryPtr(const Qnn_Tensor_t& tensor) -> void* {
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

inline auto getQnnTensorId(const Qnn_Tensor_t& tensor) -> uint32_t {
    auto tensorVariant = getTensorTypeVariant(tensor);
    return std::visit(
        [](auto&& tensorVersion) { return tensorVersion.get().id; },
        tensorVariant);
}

inline auto getQnnTensorName(const Qnn_Tensor_t& tensor) -> const char* {
    auto tensorVariant = getTensorTypeVariant(tensor);
    return std::visit(
        [](auto&& tensorVersion) { return tensorVersion.get().name; },
        tensorVariant);
}

inline auto getQnnTensorType(const Qnn_Tensor_t& tensor) -> Qnn_TensorType_t {
    auto tensorVariant = getTensorTypeVariant(tensor);
    return std::visit(
        [](auto&& tensorVersion) { return tensorVersion.get().type; },
        tensorVariant);
}

inline auto getQnnTensorDataFormat(const Qnn_Tensor_t& tensor)
    -> Qnn_TensorDataFormat_t {
    auto tensorVariant = getTensorTypeVariant(tensor);
    return std::visit(
        [](auto&& tensorVersion) { return tensorVersion.get().dataFormat; },
        tensorVariant);
}

inline auto getQnnTensorDataType(const Qnn_Tensor_t& tensor) -> Qnn_DataType_t {
    auto tensorVariant = getTensorTypeVariant(tensor);
    return std::visit(
        [](auto&& tensorVersion) { return tensorVersion.get().dataType; },
        tensorVariant);
}

inline auto getQnnTensorQuantParams(const Qnn_Tensor_t& tensor)
    -> Qnn_QuantizeParams_t {
    auto tensorVariant = getTensorTypeVariant(tensor);
    return std::visit(
        [](auto&& tensorVersion) { return tensorVersion.get().quantizeParams; },
        tensorVariant);
}

inline auto getQnnTensorRank(const Qnn_Tensor_t& tensor) -> uint32_t {
    auto tensorVariant = getTensorTypeVariant(tensor);
    return std::visit(
        [](auto&& tensorVersion) { return tensorVersion.get().rank; },
        tensorVariant);
}

inline auto getQnnTensorDimensions(const Qnn_Tensor_t& tensor) -> uint32_t* {
    auto tensorVariant = getTensorTypeVariant(tensor);
    return std::visit(
        [](auto&& tensorVersion) { return tensorVersion.get().dimensions; },
        tensorVariant);
}

inline auto getQnnTensorIsDynamicDimensions(const Qnn_Tensor_t& tensor)
    -> uint8_t* {
    if (tensor.version == QNN_TENSOR_VERSION_2) {
        return tensor.v2 /* NOLINT */.isDynamicDimensions;
    }
    return nullptr;
}

inline auto getQnnTensorSparseParams(const Qnn_Tensor_t& tensor)
    -> Qnn_SparseParams_t {
    if (tensor.version == QNN_TENSOR_VERSION_2) {
        return tensor.v2 /* NOLINT */.sparseParams;
    }
    return QNN_SPARSE_PARAMS_INIT;
}

inline auto getQnnTensorMemType(const Qnn_Tensor_t& tensor)
    -> Qnn_TensorMemType_t {
    auto tensorVariant = getTensorTypeVariant(tensor);
    return std::visit(
        [](auto&& tensorVersion) { return tensorVersion.get().memType; },
        tensorVariant);
}

inline auto getQnnTensorClientBuf(const Qnn_Tensor_t& tensor)
    -> Qnn_ClientBuffer_t {
    auto tensorVariant = getTensorTypeVariant(tensor);
    return std::visit(
        [](auto&& tensorVersion) {
            return tensorVersion.get().clientBuf /* NOLINT */;
        },
        tensorVariant);
}

inline auto getQnnTensorMemHandle(const Qnn_Tensor_t& tensor)
    -> Qnn_MemHandle_t {
    auto tensorVariant = getTensorTypeVariant(tensor);
    return std::visit(
        [](auto&& tensorVersion) {
            return tensorVersion.get().memHandle /* NOLINT */;
        },
        tensorVariant);
}

inline void setQnnTensorId(Qnn_Tensor_t& tensor, const uint32_t tensorId) {
    auto tensorVariant = getTensorTypeVariant(tensor);
    std::visit(
        [tensorId](auto&& tensorVersion) { tensorVersion.get().id = tensorId; },
        tensorVariant);
}

inline void setQnnTensorName(Qnn_Tensor_t& tensor, const char* const name) {
    auto tensorVariant = getTensorTypeVariant(tensor);
    std::visit(
        [name](auto&& tensorVersion) { tensorVersion.get().name = name; },
        tensorVariant);
}

inline void setQnnTensorType(Qnn_Tensor_t& tensor, Qnn_TensorType_t type) {
    auto tensorVariant = getTensorTypeVariant(tensor);
    std::visit(
        [type](auto&& tensorVersion) { tensorVersion.get().type = type; },
        tensorVariant);
}

inline void setQnnTensorDataFormat(Qnn_Tensor_t& tensor,
                                   Qnn_TensorDataFormat_t dataFormat) {
    auto tensorVariant = getTensorTypeVariant(tensor);
    std::visit(
        [dataFormat](auto&& tensorVersion) {
            tensorVersion.get().dataFormat = dataFormat;
        },
        tensorVariant);
}

inline void setQnnTensorDataType(Qnn_Tensor_t& tensor,
                                 const Qnn_DataType_t dataType) {
    auto tensorVariant = getTensorTypeVariant(tensor);
    std::visit(
        [dataType](auto&& tensorVersion) {
            tensorVersion.get().dataType = dataType;
        },
        tensorVariant);
}

inline void setQnnTensorQuantParams(Qnn_Tensor_t& tensor,
                                    const Qnn_QuantizeParams_t quantizeParams) {
    auto tensorVariant = getTensorTypeVariant(tensor);
    std::visit(
        [quantizeParams](auto&& tensorVersion) {
            tensorVersion.get().quantizeParams = quantizeParams;
        },
        tensorVariant);
}

inline void setQnnTensorRank(Qnn_Tensor_t& tensor, const uint32_t rank) {
    auto tensorVariant = getTensorTypeVariant(tensor);
    std::visit(
        [rank](auto&& tensorVersion) { tensorVersion.get().rank = rank; },
        tensorVariant);
}

inline void setQnnTensorDimensions(Qnn_Tensor_t& tensor,
                                   uint32_t* const dimensions /* NOLINT */) {
    auto tensorVariant = getTensorTypeVariant(tensor);
    std::visit(
        [dimensions](auto&& tensorVersion) {
            tensorVersion.get().dimensions = dimensions;
        },
        tensorVariant);
}

inline void setQnnTensorIsDynamicDimensions(
    Qnn_Tensor_t& tensor, uint8_t* const isDynamicDimensions) {
    if (tensor.version == QNN_TENSOR_VERSION_2) {
        tensor.v2 /* NOLINT */.isDynamicDimensions = isDynamicDimensions;
    }
}

inline void setQnnTensorSparseParams(Qnn_Tensor_t& tensor,
                                     const Qnn_SparseParams_t sparseParams) {
    if (tensor.version == QNN_TENSOR_VERSION_2) {
        tensor.v2 /* NOLINT */.sparseParams = sparseParams;
    }
}

inline void setQnnTensorMemType(Qnn_Tensor_t& tensor,
                                Qnn_TensorMemType_t memType) {
    auto tensorVariant = getTensorTypeVariant(tensor);
    std::visit(
        [memType](auto&& tensorVersion) {
            tensorVersion.get().memType = memType;
        },
        tensorVariant);
}

inline void setQnnTensorClientBuf(Qnn_Tensor_t& tensor,
                                  const Qnn_ClientBuffer_t clientBuf) {
    auto tensorVariant = getTensorTypeVariant(tensor);
    std::visit(
        [clientBuf](auto&& tensorVersion) {
            tensorVersion.get().clientBuf /* NOLINT */ = clientBuf;
        },
        tensorVariant);
}

inline void setQnnTensorMemHandle(Qnn_Tensor_t& tensor,
                                  const Qnn_MemHandle_t memHandle) {
    auto tensorVariant = getTensorTypeVariant(tensor);
    std::visit(
        [memHandle](auto&& tensorVersion) {
            tensorVersion.get().memHandle /* NOLINT */ = memHandle;
        },
        tensorVariant);
}

inline void freeQnnTensor(Qnn_Tensor_t& tensor) {
    /* NOLINTBEGIN */
    free((void*)getQnnTensorName(tensor));
    free(getQnnTensorDimensions(tensor));
    if (getQnnTensorIsDynamicDimensions(tensor)) {
        free(getQnnTensorIsDynamicDimensions(tensor));
    }
    /* NOLINTEND */
}

inline void freeQnnTensors(Qnn_Tensor_t*& tensors, uint32_t numTensors) {
    /* NOLINTBEGIN */
    for (size_t i = 0; i < numTensors; i++) {
        freeQnnTensor(tensors[i]);
    }
    free(tensors);
    /* NOLINTEND */
}

}  // namespace edge::qnn
