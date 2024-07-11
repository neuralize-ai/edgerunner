#include "edgerunner/qnn/backend/backend.h"

#include <CPU/QnnCpuCommon.h>
#include <GPU/QnnGpuCommon.h>
#include <HTP/QnnHtpCommon.h>
#include <dlfcn.h>
#include <fmt/core.h>

#include "edgerunner/model.hpp"

namespace edge::qnn::backend {

using QnnInterfaceGetProvidersFnT =
    Qnn_ErrorHandle_t (*)(const QnnInterface_t***, uint32_t*);

using QnnSystemInterfaceGetProvidersFnT =
    Qnn_ErrorHandle_t (*)(const QnnSystemInterface_t***, uint32_t*);

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

auto Backend::validateBackendId(const uint32_t backendId) -> STATUS {
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
