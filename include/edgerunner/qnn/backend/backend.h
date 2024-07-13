#pragma once

#include <unordered_map>

#include <HTP/QnnHtpDevice.h>
#include <QnnCommon.h>
#include <QnnInterface.h>
#include <QnnTypes.h>
#include <System/QnnSystemInterface.h>

/* TODO: move STATUS to dedicated header */
#include "edgerunner/model.hpp"

namespace edge::qnn {

class Backend {
  public:
    explicit Backend(DELEGATE delegate);

    Backend(const Backend&) = default;
    Backend(Backend&&) = delete;
    auto operator=(const Backend&) -> Backend& = delete;
    auto operator=(Backend&&) -> Backend& = delete;

    ~Backend();

    static void logCallback(const char* fmt,
                            QnnLog_Level_t level,
                            uint64_t timestamp,
                            va_list argp);

  private:
    auto loadBackend() -> STATUS;

    auto createLogger() -> STATUS;

    auto initializeBackend() -> STATUS;

    auto loadSystemLibrary() -> STATUS;

    auto createDevice() -> STATUS;

    auto createContext() -> STATUS;

    auto loadContextFromBinary() -> STATUS;

    auto validateBackendId(uint32_t backendId) const -> STATUS;

    void* m_backendLibHandle {};
    void* m_systemLibHandle {};

    Qnn_BackendHandle_t m_backendHandle {};
    QnnBackend_Config_t** m_backendConfig {};

    Qnn_DeviceHandle_t m_deviceHandle {};

    Qnn_ContextHandle_t m_context {};

    Qnn_LogHandle_t m_logHandle {};

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

}  // namespace edge::qnn
