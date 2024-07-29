#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <functional>
#include <variant>

#include "edgerunner/qnn/graph.hpp"

#include <HTP/QnnHtpContext.h>
#include <QnnCommon.h>
#include <QnnInterface.h>
#include <QnnLog.h>
#include <System/QnnSystemCommon.h>
#include <System/QnnSystemContext.h>
#include <System/QnnSystemInterface.h>
#include <dlfcn.h>
#include <fmt/core.h>
#include <nonstd/span.hpp>

#include "edgerunner/model.hpp"
#include "edgerunner/qnn/config.hpp"
#include "edgerunner/qnn/tensorOps.hpp"

namespace edge::qnn {

using QnnSystemInterfaceGetProvidersFnT =
    Qnn_ErrorHandle_t (*)(const QnnSystemInterface_t***, uint32_t*);

using ContextBinaryInfoVariant =
    std::variant<std::reference_wrapper<QnnSystemContext_BinaryInfoV1_t>,
                 std::reference_wrapper<QnnSystemContext_BinaryInfoV2_t>>;

using ConstContextBinaryInfoVariant =
    std::variant<std::reference_wrapper<const QnnSystemContext_BinaryInfoV1_t>,
                 std::reference_wrapper<const QnnSystemContext_BinaryInfoV2_t>>;

inline auto getContextBinaryInfoVariant(
    QnnSystemContext_BinaryInfo_t* contextBinaryInfo)
    -> ContextBinaryInfoVariant {
    switch (contextBinaryInfo->version) {
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1:
            return std::ref(
                contextBinaryInfo->contextBinaryInfoV1 /* NOLINT */);
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2:
            return std::ref(
                contextBinaryInfo->contextBinaryInfoV2 /* NOLINT */);
        default:
            return std::ref(
                contextBinaryInfo->contextBinaryInfoV1 /* NOLINT */);
    }
}

inline auto getContextBinaryInfoVariant(
    const QnnSystemContext_BinaryInfo_t* contextBinaryInfo)
    -> ConstContextBinaryInfoVariant {
    switch (contextBinaryInfo->version) {
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1:
            return std::cref(
                contextBinaryInfo->contextBinaryInfoV1 /* NOLINT */);
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2:
            return std::cref(
                contextBinaryInfo->contextBinaryInfoV2 /* NOLINT */);
        default:
            return std::cref(
                contextBinaryInfo->contextBinaryInfoV1 /* NOLINT */);
    }
}

GraphsInfo::~GraphsInfo() {
    if (m_graphsInfo != nullptr && m_freeGraphInfoFnHandle != nullptr) {
        m_freeGraphInfoFnHandle(&m_graphsInfo, m_graphsCount);
    } else {
        try {
            for (auto& tensor : m_inputTensors) {
                freeQnnTensor(tensor);
            }
            for (auto& tensor : m_outputTensors) {
                freeQnnTensor(tensor);
            }
        } catch (std::exception& ex) {
            fmt::print(stderr, "Failed to free graph tensors: {}\n", ex.what());
        }
    }
    if (m_context != nullptr && m_qnnInterface.contextFree != nullptr) {
        m_qnnInterface.contextFree(m_context, nullptr);
    }

    if (m_libModelHandle != nullptr) {
        try {
            dlclose(m_libModelHandle);
        } catch (std::exception& ex) {
            fmt::print(stderr,
                       "Failed to close model library handle: {}\n",
                       ex.what());
        }
    }
}

auto GraphsInfo::createContext(QNN_INTERFACE_VER_TYPE& qnnInterface,
                               Qnn_BackendHandle_t& backendHandle,
                               Qnn_DeviceHandle_t& deviceHandle) -> STATUS {
    Config<QnnContext_Config_t, void*> contextConfig {QNN_CONTEXT_CONFIG_INIT,
                                                      {}};

    m_qnnInterface = qnnInterface;

    const auto status = m_qnnInterface.contextCreate(
        backendHandle, deviceHandle, contextConfig.getPtr(), &m_context);

    if (QNN_CONTEXT_NO_ERROR != status) {
        return STATUS::FAIL;
    }

    return STATUS::SUCCESS;
}

auto GraphsInfo::loadFromSharedLibrary(const std::filesystem::path& modelPath)
    -> STATUS {
    m_libModelHandle = dlopen(modelPath.string().data(), RTLD_NOW | RTLD_LOCAL);

    if (nullptr == m_libModelHandle) {
        return STATUS::FAIL;
    }

    auto status = setComposeGraphsFnHandle(
        reinterpret_cast<ComposeGraphsFnHandleTypeT> /* NOLINT */ (
            dlsym(m_libModelHandle, "QnnModel_composeGraphs")));

    if (status == STATUS::FAIL) {
        return status;
    }

    status = setFreeGraphInfoFnHandle(
        reinterpret_cast<FreeGraphInfoFnHandleTypeT> /* NOLINT */ (
            dlsym(m_libModelHandle, "QnnModel_freeGraphsInfo")));

    return status;
}

auto GraphsInfo::setComposeGraphsFnHandle(
    ComposeGraphsFnHandleTypeT composeGraphsFnHandle) -> STATUS {
    m_composeGraphsFnHandle = composeGraphsFnHandle;

    if (m_composeGraphsFnHandle == nullptr) {
        return STATUS::FAIL;
    }

    return STATUS::SUCCESS;
}

auto GraphsInfo::setFreeGraphInfoFnHandle(
    FreeGraphInfoFnHandleTypeT freeGraphInfoFnHandle) -> STATUS {
    m_freeGraphInfoFnHandle = freeGraphInfoFnHandle;

    if (m_freeGraphInfoFnHandle == nullptr) {
        return STATUS::FAIL;
    }

    return STATUS::SUCCESS;
}

auto GraphsInfo::composeGraphs(Qnn_BackendHandle_t& qnnBackendHandle)
    -> STATUS {
    const auto status = m_composeGraphsFnHandle(qnnBackendHandle,
                                                m_qnnInterface,
                                                m_context,
                                                nullptr,
                                                0,
                                                &m_graphsInfo,
                                                &m_graphsCount,
                                                false,
                                                nullptr,
                                                QNN_LOG_LEVEL_ERROR);

    if (status != GRAPH_NO_ERROR) {
        return STATUS::FAIL;
    }

    setGraph();

    return STATUS::SUCCESS;
}

auto GraphsInfo::loadSystemLibrary() -> STATUS {
    void* systemLibraryHandle =
        dlopen("libQnnSystem.so", RTLD_NOW | RTLD_LOCAL);
    if (nullptr == systemLibraryHandle) {
        return STATUS::FAIL;
    }

    QnnSystemInterfaceGetProvidersFnT getSystemInterfaceProviders {nullptr};
    getSystemInterfaceProviders =
        reinterpret_cast<QnnSystemInterfaceGetProvidersFnT> /* NOLINT */ (
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

    const nonstd::span<QnnSystemInterface_t*> systemInterfaceProviders {
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

    -> STATUS {
    return STATUS::SUCCESS;
}

auto GraphsInfo::retrieveGraphFromContext() -> STATUS {
    for (size_t graphIdx = 0; graphIdx < m_graphsCount; ++graphIdx) {
        if (nullptr == m_qnnInterface.graphRetrieve) {
            return STATUS::FAIL;
        }
        auto& graphInfo = (*m_graphsInfo)[graphIdx] /* NOLINT */;
        if (QNN_SUCCESS
            != m_qnnInterface.graphRetrieve(
                m_context, graphInfo.graphName, &graphInfo.graph))
        {
            return STATUS::FAIL;
        }
    }

    setGraph();

    return STATUS::SUCCESS;
}

auto GraphsInfo::copyGraphsInfoV1(
    const QnnSystemContext_GraphInfoV1_t* graphInfoSrc,
    GraphInfoT* graphInfoDst) -> bool {
    graphInfoDst->graphName = nullptr;
    if (graphInfoSrc->graphName != nullptr) {
        graphInfoDst->graphName =
            strndup(graphInfoSrc->graphName, strlen(graphInfoSrc->graphName));
    }
    graphInfoDst->inputTensors = nullptr;
    graphInfoDst->numInputTensors = 0;
    if (graphInfoSrc->graphInputs != nullptr) {
        m_inputTensors = createTensorsFromInfo(graphInfoSrc->graphInputs,
                                               graphInfoSrc->numGraphInputs);
        graphInfoDst->inputTensors = m_inputTensors.data();
        graphInfoDst->numInputTensors =
            static_cast<uint32_t>(m_inputTensors.size());
    }
    graphInfoDst->outputTensors = nullptr;
    graphInfoDst->numOutputTensors = 0;
    if (graphInfoSrc->graphOutputs != nullptr) {
        m_outputTensors = createTensorsFromInfo(graphInfoSrc->graphOutputs,
                                                graphInfoSrc->numGraphOutputs);
        graphInfoDst->outputTensors = m_outputTensors.data();
        graphInfoDst->numOutputTensors =
            static_cast<uint32_t>(m_outputTensors.size());
    }
    return true;
}

auto GraphsInfo::copyGraphsInfo(const QnnSystemContext_GraphInfo_t* graphsInput,
                                const uint32_t numGraphs) -> bool {
    if (graphsInput == nullptr) {
        return false;
    }
    m_graphs.resize(numGraphs);
    m_graphPtrs.clear();
    m_graphPtrs.reserve(numGraphs);

    for (auto& graph : m_graphs) {
        m_graphPtrs.push_back(&graph);
    }

    m_graphsInfo = m_graphPtrs.data();

    const nonstd::span<const QnnSystemContext_GraphInfo_t> srcGraphs {
        graphsInput, numGraphs};

    for (uint32_t i = 0; i < numGraphs; ++i) {
        if (srcGraphs[i].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
            if (!copyGraphsInfoV1(&srcGraphs[i].graphInfoV1 /* NOLINT */,
                                  m_graphPtrs[i]))
            {
                return false;
            }
        }
    }

    return true;
}

auto GraphsInfo::copyMetadataToGraphsInfo(
    const QnnSystemContext_BinaryInfo_t* binaryInfo) -> bool {
    if (nullptr == binaryInfo) {
        return false;
    }

    m_graphsCount = 0;
    auto binaryInfoVariant = getContextBinaryInfoVariant(binaryInfo);

    return std::visit(
        [this](auto&& typedBinaryInfo) {
            if (!copyGraphsInfo(typedBinaryInfo.get().graphs,
                                typedBinaryInfo.get().numGraphs))
            {
                return false;
            }
            m_graphsCount = typedBinaryInfo.get().numGraphs;
            return true;
        },
        binaryInfoVariant

    );
}

}  // namespace edge::qnn
