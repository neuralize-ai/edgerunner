#include <vector>

#include <edgerunner/edgerunner_export.hpp>
#include <tensorflow/lite/core/c/c_api_types.h>
#include <tensorflow/lite/core/c/common.h>

#include "edgerunner/tensor.hpp"

namespace edge::tflite {

class EDGERUNNER_EXPORT TensorImpl final : public Tensor {
  public:
    explicit TensorImpl(TfLiteTensor* tfLiteTensor = nullptr)
        : m_tensor(tfLiteTensor) {}

    TensorImpl(const TensorImpl& other) = default;
    TensorImpl(TensorImpl&&) = default;
    auto operator=(const TensorImpl&) -> TensorImpl& = default;
    auto operator=(TensorImpl&&) -> TensorImpl& = default;

    ~TensorImpl() final = default;

    auto getName() const -> std::string final;

    auto getType() const -> TensorType final;

    auto getDimensions() const -> std::vector<size_t> final;

    auto getSize() const -> size_t final;

  protected:
    auto getDataPtr() -> void* final;

    auto getNumBytes() -> size_t final;

  private:
    EDGERUNNER_SUPPRESS_C4251
    TfLiteTensor* m_tensor;
};

}  // namespace edge::tflite
