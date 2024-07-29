#include <array>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <string>

#include "edgerunner/qnn/backend.hpp"

#include <CPU/QnnCpuCommon.h>
#include <GPU/QnnGpuCommon.h>
#include <HTP/QnnHtpCommon.h>
#include <HTP/QnnHtpDevice.h>
#include <HTP/QnnHtpPerfInfrastructure.h>
#include <QnnBackend.h>
#include <QnnCommon.h>
#include <QnnDevice.h>
#include <QnnInterface.h>
#include <QnnLog.h>
#include <QnnProperty.h>
#include <dlfcn.h>
#include <nonstd/span.hpp>

#include "edgerunner/model.hpp"
#include "edgerunner/qnn/config.hpp"

namespace edge::qnn {

using QnnInterfaceGetProvidersFnT =
    Qnn_ErrorHandle_t (*)(const QnnInterface_t***, uint32_t*);

Backend::Backend(const DELEGATE delegate)
    : m_delegate(delegate) {
    setCreationStatus(loadBackend());
    setCreationStatus(createLogger());
    setCreationStatus(initializeBackend());
    setCreationStatus(createDevice());
    if (delegate == DELEGATE::NPU) {
        setCreationStatus(setPowerConfig());
    }
}

Backend::~Backend() {
    destroyPowerConfig();

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

    auto getInterfaceProviders =
        reinterpret_cast<QnnInterfaceGetProvidersFnT> /* NOLINT */ (
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

    const nonstd::span<QnnInterface_t*> interfaceProviders {
        interfaceProvidersPtr, numProviders};

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

void Backend::logCallback(const char* fmtStr,
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

    /* NOLINTBEGIN */
    std::fprintf(stderr, "%8.1lums [%-7s] ", timestamp, levelStr.c_str());
    std::vfprintf(stderr, fmtStr, argp);
    std::fprintf(stderr, "\n");
    /* NOLINTEND */
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
        m_logHandle,
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

    Config<QnnDevice_Config_t, void*> deviceConfig {QNN_DEVICE_CONFIG_INIT, {}};

    if (nullptr != m_qnnInterface.deviceCreate) {
        auto qnnStatus = m_qnnInterface.deviceCreate(
            m_logHandle, deviceConfig.getPtr(), &m_deviceHandle);
        if (QNN_SUCCESS != qnnStatus) {
            return STATUS::FAIL;
        }
    } else {
        return STATUS::FAIL;
    }

    return STATUS::SUCCESS;
}

auto Backend::setPowerConfig() -> STATUS {
    if (m_delegate != DELEGATE::NPU) {
        return STATUS::FAIL;
    }

    QnnDevice_Infrastructure_t deviceInfrastructure = nullptr;
    if (QNN_SUCCESS
        != m_qnnInterface.deviceGetInfrastructure(&deviceInfrastructure))
    {
        return STATUS::FAIL;
    }

    const auto* htpDeviceInfraStructure =
        static_cast<QnnHtpDevice_Infrastructure_t*>(deviceInfrastructure);
    m_devicePerfInfrastructure = htpDeviceInfraStructure->perfInfra;  // NOLINT
    ;
    if (QNN_SUCCESS
        != m_devicePerfInfrastructure.createPowerConfigId(
            0, 0, &m_powerConfigId))
    {
        return STATUS::FAIL;
    }

    QnnHtpPerfInfrastructure_PowerConfig_t powerConfig =
        QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;

    powerConfig.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;

    auto& dcvsConfig = powerConfig.dcvsV3Config;  // NOLINT

    dcvsConfig.dcvsEnable = 0;
    dcvsConfig.setDcvsEnable = 1;
    dcvsConfig.contextId = m_powerConfigId;

    dcvsConfig.powerMode =
        QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
    dcvsConfig.setSleepLatency = 1;
    dcvsConfig.setBusParams = 1;
    dcvsConfig.setCoreParams = 1;
    dcvsConfig.sleepDisable = 1;
    dcvsConfig.setSleepDisable = 1;

    /* 10-65535 us */
    static constexpr uint32_t SleepLatency = 40;
    dcvsConfig.sleepLatency = SleepLatency;

    // set Bus Clock Parameters (refer QnnHtpPerfInfrastructure.h)
    dcvsConfig.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    dcvsConfig.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    dcvsConfig.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;

    // set Core Clock Parameters (refer QnnHtpPerfInfrastructure.h)
    dcvsConfig.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    dcvsConfig.coreVoltageCornerTarget =
        DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    dcvsConfig.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;

    // Set power config with different performance parameters
    std::array<const QnnHtpPerfInfrastructure_PowerConfig_t*, 2> powerConfigs =
        {&powerConfig, NULL};

    if (QNN_SUCCESS
        != m_devicePerfInfrastructure.setPowerConfig(m_powerConfigId,
                                                     powerConfigs.data()))
    {
        return STATUS::FAIL;
    }

    return STATUS::SUCCESS;
}

auto Backend::destroyPowerConfig() const -> STATUS {
    if (QNN_SUCCESS
        != m_devicePerfInfrastructure.destroyPowerConfigId(m_powerConfigId))
    {
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
