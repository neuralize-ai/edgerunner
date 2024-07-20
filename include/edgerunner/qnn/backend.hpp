/**
 * @file Backend.h
 * @brief Definition of the Backend class for handling QNN backends.
 *
 * This class represents a backend for handling interfacing with QNN backend
 * libraries. It provides functionality for loading the backend, creating a
 * device, initializing the backend, and managing the context for QNN
 * operations.
 *
 * The Backend class is currently restricted to NPU inference support.
 */

#pragma once

#include <unordered_map>

#include <HTP/QnnHtpDevice.h>
#include <QnnCommon.h>
#include <QnnInterface.h>
#include <QnnTypes.h>
#include <System/QnnSystemInterface.h>
#include <fmt/core.h>

/* TODO: move STATUS to dedicated header */
#include "edgerunner/model.hpp"

namespace edge::qnn {

/**
 * @class Backend
 * @brief Class for handling QNN backends.
 */
class Backend {
  public:
    /**
     * @brief Constructor for the Backend class.
     * @param delegate The delegate type for the backend (CPU, GPU, NPU).
     * @param isContextBinary Whether the model will be loaded from a context
     * binary.
     */
    explicit Backend(DELEGATE delegate, bool isContextBinary);

    Backend(const Backend&) = default;
    Backend(Backend&&) = delete;
    auto operator=(const Backend&) -> Backend& = delete;
    auto operator=(Backend&&) -> Backend& = delete;

    /**
     * @brief Destructor for the Backend class.
     */
    ~Backend();

    /**
     * @brief Get the backend handle.
     * @return Reference to the backend handle.
     */
    auto getHandle() -> auto& { return m_backendHandle; }

    /**
     * @brief Get the context for the backend.
     * @return Reference to the backend context.
     */
    auto getContext() -> auto& { return m_context; }

    /**
     * @brief Get the QNN interface.
     * @return Reference to the QNN interface.
     */
    auto getInterface() -> auto& { return m_qnnInterface; }

    /**
     * @brief Get the QNN system interface.
     * @return Reference to the QNN system interface.
     */
    auto getSystemInterface() -> auto& { return m_qnnSystemInterface; }

    /**
     * @brief Get the delegate type for the backend.
     * @return The delegate type.
     */
    auto getDelegate() { return m_delegate; }

    /**
     * @brief Static callback function for logging.
     * @param fmtStr The format string for the log message.
     * @param level The log level.
     * @param timestamp The timestamp of the log message.
     * @param argp Additional arguments for the log message.
     */
    static void logCallback(const char* fmtStr,
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

    auto setPowerConfig() -> STATUS;

    auto destroyPowerConfig() const -> STATUS;

    auto loadContextFromBinary() -> STATUS;

    auto validateBackendId(uint32_t backendId) const -> STATUS;

    void* m_backendLibHandle {};
    void* m_systemLibHandle {};

    Qnn_BackendHandle_t m_backendHandle {};
    QnnBackend_Config_t** m_backendConfig {};

    Qnn_DeviceHandle_t m_deviceHandle {};

    Qnn_ContextHandle_t m_context {};

    Qnn_LogHandle_t m_logHandle {};

    uint32_t m_powerConfigId {};

    QnnHtpDevice_PerfInfrastructure_t m_devicePerfInfrastructure {};

    QNN_INTERFACE_VER_TYPE m_qnnInterface = QNN_INTERFACE_VER_TYPE_INIT;
    QNN_SYSTEM_INTERFACE_VER_TYPE m_qnnSystemInterface =
        QNN_SYSTEM_INTERFACE_VER_TYPE_INIT;

    DELEGATE m_delegate;

    std::unordered_map<DELEGATE, std::string> m_backendLibrariesByDelegate {
        {DELEGATE::CPU, "libQnnCpu.so"},
        {DELEGATE::GPU, "libQnnGpu.so"},
        {DELEGATE::NPU, "libQnnHtp.so"}};

    uint32_t m_deviceId {};
    QnnHtpDevice_Arch_t m_htpArch {};
};

}  // namespace edge::qnn
