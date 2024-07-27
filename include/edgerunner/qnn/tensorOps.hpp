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

auto getTensorTypeVariant(Qnn_Tensor_t& tensor) -> TensorVariant;

auto getTensorTypeVariant(const Qnn_Tensor_t& tensor) -> ConstTensorVariant;

using TensorMemoryVariant =
    std::variant<Qnn_MemHandle_t, std::reference_wrapper<Qnn_ClientBuffer_t>>;

auto getTensorMemoryPtr(const Qnn_Tensor_t& tensor) -> void*;

auto getQnnTensorId(const Qnn_Tensor_t& tensor) -> uint32_t;

auto getQnnTensorName(const Qnn_Tensor_t& tensor) -> const char*;

auto getQnnTensorType(const Qnn_Tensor_t& tensor) -> Qnn_TensorType_t;

auto getQnnTensorDataFormat(const Qnn_Tensor_t& tensor)
    -> Qnn_TensorDataFormat_t;

auto getQnnTensorDataType(const Qnn_Tensor_t& tensor) -> Qnn_DataType_t;

auto getQnnTensorQuantParams(const Qnn_Tensor_t& tensor)
    -> Qnn_QuantizeParams_t;

auto getQnnTensorRank(const Qnn_Tensor_t& tensor) -> uint32_t;

auto getQnnTensorDimensions(const Qnn_Tensor_t& tensor) -> uint32_t*;

auto getQnnTensorIsDynamicDimensions(const Qnn_Tensor_t& tensor) -> uint8_t*;

auto getQnnTensorSparseParams(const Qnn_Tensor_t& tensor) -> Qnn_SparseParams_t;

auto getQnnTensorMemType(const Qnn_Tensor_t& tensor) -> Qnn_TensorMemType_t;

auto getQnnTensorClientBuf(const Qnn_Tensor_t& tensor) -> Qnn_ClientBuffer_t;

auto getQnnTensorMemHandle(const Qnn_Tensor_t& tensor) -> Qnn_MemHandle_t;

void setQnnTensorId(Qnn_Tensor_t& tensor, uint32_t tensorId);

void setQnnTensorName(Qnn_Tensor_t& tensor, const char* name);

void setQnnTensorType(Qnn_Tensor_t& tensor, Qnn_TensorType_t type);

void setQnnTensorDataFormat(Qnn_Tensor_t& tensor,
                            Qnn_TensorDataFormat_t dataFormat);

void setQnnTensorDataType(Qnn_Tensor_t& tensor, Qnn_DataType_t dataType);

void setQnnTensorQuantParams(Qnn_Tensor_t& tensor,
                             Qnn_QuantizeParams_t quantizeParams);

void setQnnTensorRank(Qnn_Tensor_t& tensor, uint32_t rank);

void setQnnTensorDimensions(Qnn_Tensor_t& tensor,
                            uint32_t* const dimensions /* NOLINT */);

void setQnnTensorIsDynamicDimensions(Qnn_Tensor_t& tensor,
                                     uint8_t* isDynamicDimensions);

void setQnnTensorSparseParams(Qnn_Tensor_t& tensor,
                              Qnn_SparseParams_t sparseParams);

void setQnnTensorMemType(Qnn_Tensor_t& tensor, Qnn_TensorMemType_t memType);

void setQnnTensorClientBuf(Qnn_Tensor_t& tensor, Qnn_ClientBuffer_t clientBuf);

void setQnnTensorMemHandle(Qnn_Tensor_t& tensor, Qnn_MemHandle_t memHandle);

void freeQnnTensor(Qnn_Tensor_t& tensor);

auto deepCopyQnnTensorInfo(Qnn_Tensor_t& dst, const Qnn_Tensor_t& src) -> bool;

auto createTensorsFromInfo(const Qnn_Tensor_t* tensorsInfoSrc,
                           uint32_t tensorsCount) -> std::vector<Qnn_Tensor_t>;

}  // namespace edge::qnn
