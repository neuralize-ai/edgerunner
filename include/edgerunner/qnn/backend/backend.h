
#include <unordered_map>
#include <vector>

#include <HTP/QnnHtpDevice.h>
#include <QnnCommon.h>
#include <QnnInterface.h>
#include <System/QnnSystemInterface.h>

#include "edgerunner/model.hpp"

namespace edge::qnn::backend {

class Backend {
  public:
    explicit Backend(DELEGATE delegate)
        : m_delegate(delegate) {}

    auto loadBackend() -> STATUS;

    auto initializeBackend() -> STATUS;

    auto createDevice() -> STATUS;

    auto createContext() -> STATUS;

    auto loadContextFromBinary() -> STATUS;

  private:
    auto loadSystemLibrary() -> STATUS;

    auto getQnnInterfaceProvider() -> STATUS;

    auto validateBackendId(uint32_t backendId) -> STATUS;

    void* m_backendLibHandle {};
    void* m_systemLibHandle {};

    Qnn_BackendHandle_t m_backendHandle {};
    QnnBackend_Config_t** m_backendConfig {};

    Qnn_DeviceHandle_t m_deviceHandle {};

    std::vector<Qnn_ContextHandle_t> m_contexts;

    QNN_INTERFACE_VER_TYPE m_qnnInterface = QNN_INTERFACE_VER_TYPE_INIT;
    QNN_SYSTEM_INTERFACE_VER_TYPE m_qnnSystemInterface =
        QNN_SYSTEM_INTERFACE_VER_TYPE_INIT;

    DELEGATE m_delegate;

    const std::unordered_map<DELEGATE, std::string>
        m_backendLibrariesByDelegate {{DELEGATE::CPU, "libQnnCpu.so"},
                                      {DELEGATE::GPU, "libQnnGpu.so"},
                                      {DELEGATE::NPU, "libQnnHtp.so"}};

    uint32_t m_deviceId {};
    QnnHtpDevice_Arch_t m_htpArch {};
};

}  // namespace edge::qnn::backend
