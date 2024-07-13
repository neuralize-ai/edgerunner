#include <cstdio>

#include "edgerunner/qnn/backend.h"

#include <CPU/QnnCpuCommon.h>
#include <GPU/QnnGpuCommon.h>
#include <HTP/QnnHtpCommon.h>
#include <HTP/QnnHtpDevice.h>
#include <QnnCommon.h>
#include <QnnContext.h>
#include <QnnLog.h>
#include <dlfcn.h>

#include "edgerunner/model.hpp"
#include "edgerunner/qnn/config.h"

namespace edge::qnn {

using QnnInterfaceGetProvidersFnT =
    Qnn_ErrorHandle_t (*)(const QnnInterface_t***, uint32_t*);

using QnnSystemInterfaceGetProvidersFnT =
    Qnn_ErrorHandle_t (*)(const QnnSystemInterface_t***, uint32_t*);

Backend::Backend(const DELEGATE delegate)
    : m_delegate(delegate) {
    loadBackend();

    createLogger();

    initializeBackend();

    createDevice();

    createContext();
}

Backend::~Backend() {
    if (m_context != nullptr && m_qnnInterface.contextFree != nullptr) {
        m_qnnInterface.contextFree(m_context, nullptr);
    }

    if (m_deviceHandle != nullptr && m_qnnInterface.deviceFree != nullptr) {
        m_qnnInterface.deviceFree(m_deviceHandle);
    }

    if (m_backendHandle != nullptr && m_qnnInterface.backendFree != nullptr) {
        m_qnnInterface.backendFree(m_backendHandle);
    }

    dlclose(m_backendLibHandle);
}

auto Backend::loadBackend() -> STATUS {
    m_backendLibHandle =
        dlopen(m_backendLibrariesByDelegate.at(m_delegate).c_str(),
               RTLD_NOW | RTLD_LOCAL);

    if (nullptr == m_backendLibHandle) {
        return STATUS::FAIL;
    }

    auto getInterfaceProviders = reinterpret_cast<QnnInterfaceGetProvidersFnT>(
        dlsym(m_backendLibHandle, "QnnInterface_getProviders"));

    if (nullptr == getInterfaceProviders) {
        dlclose(m_backendLibHandle);
        return STATUS::FAIL;
    }

    QnnInterface_t** interfaceProvidersPtr {};
    uint32_t numProviders {};

    if (QNN_SUCCESS
        != getInterfaceProviders(
            const_cast<const QnnInterface_t***>(&interfaceProvidersPtr),
            &numProviders))
    {
        dlclose(m_backendLibHandle);
        return STATUS::FAIL;
    }
    if (nullptr == interfaceProvidersPtr || 0 == numProviders) {
        dlclose(m_backendLibHandle);
        return STATUS::FAIL;
    }

    nonstd::span<QnnInterface_t*> interfaceProviders {interfaceProvidersPtr,
                                                      numProviders};

    uint32_t backendId = 0;
    for (const auto& interfaceProvider : interfaceProviders) {
        const auto& coreApiVersion =
            interfaceProvider->apiVersion.coreApiVersion;
        if (QNN_API_VERSION_MAJOR == coreApiVersion.major
            && QNN_API_VERSION_MINOR <= coreApiVersion.minor)
        {
            m_qnnInterface = interfaceProvider->QNN_INTERFACE_VER_NAME;
            backendId = interfaceProvider->backendId;
        } else {
            dlclose(m_backendLibHandle);
            return STATUS::FAIL;
        }
    }

    return validateBackendId(backendId);
}

auto Backend::loadSystemLibrary() -> STATUS {
    void* systemLibraryHandle =
        dlopen("libQnnSystem.so", RTLD_NOW | RTLD_LOCAL);
    if (nullptr == systemLibraryHandle) {
        return STATUS::FAIL;
    }

    QnnSystemInterfaceGetProvidersFnT getSystemInterfaceProviders {nullptr};
    getSystemInterfaceProviders =
        reinterpret_cast<QnnSystemInterfaceGetProvidersFnT>(
            dlsym(systemLibraryHandle, "QnnSystemInterface_getProviders"));
    if (nullptr == getSystemInterfaceProviders) {
        return STATUS::FAIL;
    }

    QnnSystemInterface_t** systemInterfaceProvidersPtr {nullptr};
    uint32_t numProviders = 0;
    if (QNN_SUCCESS
        != getSystemInterfaceProviders(
            const_cast<const QnnSystemInterface_t***>(
                &systemInterfaceProvidersPtr),
            &numProviders))
    {
        return STATUS::FAIL;
    }
    if (nullptr == systemInterfaceProvidersPtr || 0 == numProviders) {
        return STATUS::FAIL;
    }

    nonstd::span<QnnSystemInterface_t*> systemInterfaceProviders {
        systemInterfaceProvidersPtr, numProviders};

    for (const auto& systemInterfaceProvider : systemInterfaceProviders) {
        const auto systemApiVersion = systemInterfaceProvider->systemApiVersion;

        if (QNN_SYSTEM_API_VERSION_MAJOR == systemApiVersion.major
            && QNN_SYSTEM_API_VERSION_MINOR <= systemApiVersion.minor)
        {
            m_qnnSystemInterface =
                systemInterfaceProvider->QNN_SYSTEM_INTERFACE_VER_NAME;
            return STATUS::SUCCESS;
        }
    }

    return STATUS::FAIL;
}

void Backend::logCallback(const char* fmt,
                          QnnLog_Level_t level,
                          uint64_t timestamp,
                          va_list argp) {
    std::string levelStr;

    switch (level) {
        case QNN_LOG_LEVEL_ERROR:
            levelStr = "ERROR";
            break;
        case QNN_LOG_LEVEL_WARN:
            levelStr = "WARNING";
            break;
        case QNN_LOG_LEVEL_INFO:
            levelStr = "INFO";
            break;
        case QNN_LOG_LEVEL_DEBUG:
            levelStr = "DEBUG";
            break;
        case QNN_LOG_LEVEL_VERBOSE:
            levelStr = "VERBOSE";
            break;
        case QNN_LOG_LEVEL_MAX:
            levelStr = "UNKNOWN";
            break;
    }

    std::fprintf(stdout, "%8.1lums [%-7s] ", timestamp, levelStr.c_str());
    std::vfprintf(stdout, fmt, argp);
    std::fprintf(stdout, "\n");
}

auto Backend::createLogger() -> STATUS {
    if (QNN_SUCCESS
        != m_qnnInterface.logCreate(
            logCallback, QNN_LOG_LEVEL_ERROR, &m_logHandle))
    {
        return STATUS::FAIL;
    }

    return STATUS::SUCCESS;
}

auto Backend::initializeBackend() -> STATUS {
    const auto status = m_qnnInterface.backendCreate(
        nullptr,
        const_cast<const QnnBackend_Config_t**>(m_backendConfig),
        &m_backendHandle);
    if (QNN_BACKEND_NO_ERROR != status) {
        return STATUS::FAIL;
    }

    return STATUS::SUCCESS;
}

auto Backend::createDevice() -> STATUS {
    const auto& propertyHasCapability = m_qnnInterface.propertyHasCapability;
    if (nullptr != propertyHasCapability) {
        auto status = propertyHasCapability(QNN_PROPERTY_GROUP_DEVICE);
        if (QNN_PROPERTY_ERROR_UNKNOWN_KEY == status) {
            return STATUS::FAIL;
        }
    }

    Config<QnnDevice_Config_t, QnnHtpDevice_CustomConfig_t> deviceConfig {
        QNN_DEVICE_CONFIG_INIT, {}};

    if (nullptr != m_qnnInterface.deviceCreate) {
        auto qnnStatus = m_qnnInterface.deviceCreate(
            nullptr, deviceConfig.getPtr(), &m_deviceHandle);
        if (QNN_SUCCESS != qnnStatus) {
            return STATUS::FAIL;
        }
    }

    return STATUS::SUCCESS;
}

auto Backend::createContext() -> STATUS {
    Config<QnnContext_Config_t, void*> contextConfig {QNN_CONTEXT_CONFIG_INIT,
                                                      {}};

    const auto status = m_qnnInterface.contextCreate(
        m_backendHandle, m_deviceHandle, contextConfig.getPtr(), &m_context);

    if (QNN_CONTEXT_NO_ERROR != status) {
        return STATUS::FAIL;
    }

    return STATUS::SUCCESS;
}

auto Backend::validateBackendId(const uint32_t backendId) const -> STATUS {
    switch (backendId) {
        case QNN_BACKEND_ID_CPU:
            return m_delegate == DELEGATE::CPU ? STATUS::SUCCESS : STATUS::FAIL;
        case QNN_BACKEND_ID_GPU:
            return m_delegate == DELEGATE::GPU ? STATUS::SUCCESS : STATUS::FAIL;
        case QNN_BACKEND_ID_HTP:
            return m_delegate == DELEGATE::NPU ? STATUS::SUCCESS : STATUS::FAIL;
        default:
            return STATUS::FAIL;
    }
}

}  // namespace edge::qnn
