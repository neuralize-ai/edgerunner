#pragma once

#include <cstdlib>
#include <cstring>
#include <functional>
#include <variant>

#include <QnnTypes.h>
#include <nonstd/span.hpp>

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
    free(std::remove_const_t<void*>(getQnnTensorName(tensor)));
    free(getQnnTensorDimensions(tensor));
    if (getQnnTensorIsDynamicDimensions(tensor)) {
        free(getQnnTensorIsDynamicDimensions(tensor));
    }
    if (getQnnTensorQuantParams(tensor).axisScaleOffsetEncoding.scaleOffset) {
        free(getQnnTensorQuantParams(tensor)
                 .axisScaleOffsetEncoding.scaleOffset);
    }
    /* NOLINTEND */
}

inline auto deepCopyQnnTensorInfo(Qnn_Tensor_t& dst,
                                  const Qnn_Tensor_t& src) -> bool {
    dst.version = src.version;
    const char* tensorName = getQnnTensorName(src);
    if (tensorName == nullptr) {
        setQnnTensorName(dst, nullptr);
    } else {
        setQnnTensorName(dst, strndup(tensorName, strlen(tensorName)));
    }
    setQnnTensorId(dst, getQnnTensorId(src));
    setQnnTensorType(dst, getQnnTensorType(src));
    setQnnTensorDataFormat(dst, getQnnTensorDataFormat(src));
    setQnnTensorDataType(dst, getQnnTensorDataType(src));
    Qnn_QuantizeParams_t qParams = QNN_QUANTIZE_PARAMS_INIT;
    qParams.encodingDefinition =
        getQnnTensorQuantParams(src).encodingDefinition;
    qParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
    if (getQnnTensorQuantParams(src).quantizationEncoding
        == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET)
    {
        qParams.quantizationEncoding =
            getQnnTensorQuantParams(src).quantizationEncoding;
        qParams.scaleOffsetEncoding /* NOLINT */ =
            getQnnTensorQuantParams(src).scaleOffsetEncoding /* NOLINT */;
    } else if (getQnnTensorQuantParams(src).quantizationEncoding
               == QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET)
    {
        qParams.quantizationEncoding =
            getQnnTensorQuantParams(src).quantizationEncoding;
        qParams.axisScaleOffsetEncoding /* NOLINT */.axis =
            getQnnTensorQuantParams(src)
                .axisScaleOffsetEncoding /* NOLINT */.axis;
        qParams.axisScaleOffsetEncoding /* NOLINT */.numScaleOffsets =
            getQnnTensorQuantParams(src)
                .axisScaleOffsetEncoding /* NOLINT */.numScaleOffsets;

        if (getQnnTensorQuantParams(src)
                .axisScaleOffsetEncoding /* NOLINT */.numScaleOffsets
            > 0)
        {
            qParams.axisScaleOffsetEncoding.scaleOffset /* NOLINT */ =
                reinterpret_cast<Qnn_ScaleOffset_t*> /* NOLINT */ (malloc(
                    getQnnTensorQuantParams(src)
                        .axisScaleOffsetEncoding /* NOLINT */.numScaleOffsets
                    * sizeof(Qnn_ScaleOffset_t)));
            if (qParams.axisScaleOffsetEncoding /* NOLINT */.scaleOffset
                != nullptr)
            {
                for (size_t idx = 0;
                     idx < getQnnTensorQuantParams(src)
                               .axisScaleOffsetEncoding /* NOLINT */
                               .numScaleOffsets;
                     idx++)
                {
                    qParams /* NOLINT */
                        .axisScaleOffsetEncoding /* NOLINT */.scaleOffset[idx]
                        .scale = getQnnTensorQuantParams(src) /* NOLINT */
                                     .axisScaleOffsetEncoding /* NOLINT */
                                     .scaleOffset[idx]
                                     .scale;
                    qParams /* NOLINT */
                        .axisScaleOffsetEncoding /* NOLINT */.scaleOffset[idx]
                        .offset = getQnnTensorQuantParams(src) /* NOLINT */
                                      .axisScaleOffsetEncoding /* NOLINT */
                                      .scaleOffset[idx]
                                      .offset;
                }
            }
        }
    }
    setQnnTensorQuantParams(dst, qParams);
    setQnnTensorRank(dst, getQnnTensorRank(src));
    setQnnTensorDimensions(dst, nullptr);
    if (getQnnTensorRank(src) > 0) {
        setQnnTensorDimensions(dst,
                               static_cast<uint32_t*>(malloc /* NOLINT */ (
                                   getQnnTensorRank(src) * sizeof(uint32_t))));
        if (getQnnTensorDimensions(dst) != nullptr) {
            memcpy(getQnnTensorDimensions(dst),
                   getQnnTensorDimensions(src),
                   getQnnTensorRank(src) * sizeof(uint32_t));
        }
        if (getQnnTensorIsDynamicDimensions(src) != nullptr) {
            setQnnTensorIsDynamicDimensions(
                dst,
                static_cast<uint8_t*>(malloc /* NOLINT */ (getQnnTensorRank(src)
                                                           * sizeof(uint8_t))));
            memcpy(getQnnTensorIsDynamicDimensions(dst),
                   getQnnTensorIsDynamicDimensions(src),
                   getQnnTensorRank(src) * sizeof(uint8_t));
        }
    }

    setQnnTensorSparseParams(dst, getQnnTensorSparseParams(src));

    return true;
}

inline auto createTensorsFromInfo(const Qnn_Tensor_t* tensorsInfoSrc,
                                  uint32_t tensorsCount)
    -> std::vector<Qnn_Tensor_t> {
    const nonstd::span<const Qnn_Tensor_t> tensorsInfo {tensorsInfoSrc,
                                                        tensorsCount};
    std::vector<Qnn_Tensor_t> tensorWrappers(tensorsCount);
    for (size_t tIdx = 0; tIdx < tensorsCount; ++tIdx) {
        tensorWrappers[tIdx] /* NOLINT */ = QNN_TENSOR_INIT;
        if (!deepCopyQnnTensorInfo(tensorWrappers[tIdx], tensorsInfo[tIdx])) {
            return {};
        }
    }

    return tensorWrappers;
}

}  // namespace edge::qnn
