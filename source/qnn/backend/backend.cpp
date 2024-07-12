#include "edgerunner/qnn/backend/backend.h"

#include <CPU/QnnCpuCommon.h>
#include <GPU/QnnGpuCommon.h>
#include <HTP/QnnHtpCommon.h>
#include <QnnContext.h>
#include <dlfcn.h>
#include <fmt/core.h>

#include "edgerunner/model.hpp"
#include "edgerunner/qnn/backend/config.h"

namespace edge::qnn::backend {

using QnnInterfaceGetProvidersFnT =
    Qnn_ErrorHandle_t (*)(const QnnInterface_t***, uint32_t*);

using QnnSystemInterfaceGetProvidersFnT =
    Qnn_ErrorHandle_t (*)(const QnnSystemInterface_t***, uint32_t*);

Backend::Backend(const DELEGATE delegate)
    : m_delegate(delegate) {
    loadBackend();

    initializeBackend();

    createDevice();

    createContext();
}

Backend::~Backend() {
    if (m_backendLibHandle != nullptr) {
        dlclose(m_backendLibHandle);
    }

    if (m_backendHandle != nullptr && m_qnnInterface.backendFree != nullptr) {
        m_qnnInterface.backendFree(m_backendHandle);
    }

    if (m_deviceHandle != nullptr && m_qnnInterface.deviceFree != nullptr) {
        m_qnnInterface.deviceFree(m_deviceHandle);
    }

    if (m_context != nullptr && m_qnnInterface.contextFree != nullptr) {
        m_qnnInterface.contextFree(m_context, nullptr);
    }
}

auto Backend::loadBackend() -> STATUS {
    m_backendLibHandle =
        dlopen(m_backendLibrariesByDelegate.at(m_delegate).c_str(),
               RTLD_NOW | RTLD_LOCAL);

    if (nullptr == m_backendLibHandle) {
        fmt::print(stderr, "load lib backend handle failed\n");
        return STATUS::FAIL;
    }

    auto getInterfaceProviders = reinterpret_cast<QnnInterfaceGetProvidersFnT>(
        dlsym(m_backendLibHandle, "QnnInterface_getProviders"));

    if (nullptr == getInterfaceProviders) {
        fmt::print(stderr, "get interface providers fn failed\n");
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
        fmt::print(stderr, "get interface providers failed\n");
        dlclose(m_backendLibHandle);
        return STATUS::FAIL;
    }
    if (nullptr == interfaceProvidersPtr || 0 == numProviders) {
        fmt::print(stderr, "get interface providers failed 2\n");
        dlclose(m_backendLibHandle);
        return STATUS::FAIL;
    }

    nonstd::span<QnnInterface_t*> interfaceProviders {interfaceProvidersPtr,
                                                      numProviders};

    uint32_t backendId;
    for (const auto& interfaceProvider : interfaceProviders) {
        const auto& coreApiVersion =
            interfaceProvider->apiVersion.coreApiVersion;
        if (QNN_API_VERSION_MAJOR == coreApiVersion.major
            && QNN_API_VERSION_MINOR <= coreApiVersion.minor)
        {
            m_qnnInterface = interfaceProvider->QNN_INTERFACE_VER_NAME;
            backendId = interfaceProvider->backendId;
        } else {
            fmt::print(stderr, "interface providers invalid\n");
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
        fmt::print(stderr, "load system library failed\n");
        return STATUS::FAIL;
    }

    QnnSystemInterfaceGetProvidersFnT getSystemInterfaceProviders {nullptr};
    getSystemInterfaceProviders =
        reinterpret_cast<QnnSystemInterfaceGetProvidersFnT>(
            dlsym(systemLibraryHandle, "QnnSystemInterface_getProviders"));
    if (nullptr == getSystemInterfaceProviders) {
        fmt::print(stderr, "get system interface providers fn failed\n");
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
        fmt::print(stderr, "get system interface providers failed\n");
        return STATUS::FAIL;
    }
    if (nullptr == systemInterfaceProvidersPtr || 0 == numProviders) {
        fmt::print(stderr, "get system interface providers failed 2\n");
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

    fmt::print(stderr, "system interface providers invalid\n");
    return STATUS::FAIL;
}

auto Backend::initializeBackend() -> STATUS {
    const auto status = m_qnnInterface.backendCreate(
        nullptr,
        const_cast<const QnnBackend_Config_t**>(m_backendConfig),
        &m_backendHandle);
    if (QNN_BACKEND_NO_ERROR != status) {
        fmt::print(stderr, "initialize backend failed\n");
        return STATUS::FAIL;
    }

    return STATUS::SUCCESS;
}

auto Backend::createDevice() -> STATUS {
    const auto& propertyHasCapability = m_qnnInterface.propertyHasCapability;
    if (nullptr != propertyHasCapability) {
        auto status = propertyHasCapability(QNN_PROPERTY_GROUP_DEVICE);
        if (QNN_PROPERTY_ERROR_UNKNOWN_KEY == status) {
            fmt::print(stderr, "device property supported failed\n");
            return STATUS::FAIL;
        }
    }

    Config<QnnDevice_Config_t, QnnHtpDevice_CustomConfig_t> deviceConfig {
        QNN_DEVICE_CONFIG_INIT, {}};

    if (nullptr != m_qnnInterface.deviceCreate) {
        auto qnnStatus = m_qnnInterface.deviceCreate(
            nullptr, deviceConfig.getPtr(), &m_deviceHandle);
        if (QNN_SUCCESS != qnnStatus) {
            fmt::print(stderr, "create device failed\n");
            return STATUS::FAIL;
        }
    }

    return STATUS::SUCCESS;
}

auto Backend::createContext() -> STATUS {
    Config<QnnContext_Config_t, QnnContext_CustomConfig_t> contextConfig {
        QNN_CONTEXT_CONFIG_INIT, {}};
    const auto status = m_qnnInterface.contextCreate(
        m_backendHandle, m_deviceHandle, contextConfig.getPtr(), &m_context);

    if (QNN_CONTEXT_NO_ERROR != status) {
        fmt::print(stderr, "create context failed\n");
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

}  // namespace edge::qnn::backend
