#include <cstdint>
#include <filesystem>
#include <memory>

#include "edgerunner/model.hpp"

#include <HTP/QnnHtpDevice.h>
#include <HTP/QnnHtpGraph.h>
#include <QnnCommon.h>
#include <QnnContext.h>
#include <QnnGraph.h>
#include <QnnInterface.h>
#include <QnnLog.h>
#include <QnnTypes.h>
#include <System/QnnSystemContext.h>
#include <System/QnnSystemInterface.h>
#include <dlfcn.h>
#include <nonstd/span.hpp>

#include "edgerunner/qnn/config.h"
#include "edgerunner/qnn/model.hpp"
#include "edgerunner/qnn/tensor.hpp"

namespace edge::qnn {

auto ModelImpl::loadModel(const std::filesystem::path& modelPath) -> STATUS {
    return loadFromSharedLibrary(modelPath);
}

auto ModelImpl::loadModel(const nonstd::span<uint8_t>& /*modelBuffer*/)
    -> STATUS {
    return STATUS::FAIL;
}

auto ModelImpl::applyDelegate(const DELEGATE& delegate) -> STATUS {
    if (delegate != DELEGATE::NPU) {
        return STATUS::FAIL;
    }
    return STATUS::SUCCESS;
}

auto ModelImpl::composeGraphs() -> STATUS {
    auto& qnnInterface = m_backend->getInterface();
    auto& qnnContext = m_backend->getContext();
    auto& qnnBackendHandle = m_backend->getHandle();

    const auto status = m_composeGraphsFnHandle(qnnBackendHandle,
                                                qnnInterface,
                                                qnnContext,
                                                nullptr,
                                                0,
                                                &m_graphInfo,
                                                &m_graphsCount,
                                                false,
                                                nullptr,
                                                QNN_LOG_LEVEL_ERROR);

    if (ModelErrorT::MODEL_NO_ERROR != status) {
        return STATUS::FAIL;
    }

    return STATUS::SUCCESS;
}

auto ModelImpl::finalizeGraphs() -> STATUS {
    auto& qnnInterface = m_backend->getInterface();
    auto& graphInfo = (*m_graphInfo)[0];

    const auto status =
        qnnInterface.graphFinalize(graphInfo.graph, nullptr, nullptr);

    if (QNN_GRAPH_NO_ERROR != status) {
        return STATUS::FAIL;
    }

    bool saveBinary = false;
    if (saveBinary) {
        /* TODO: save binary */
    }

    return STATUS::SUCCESS;
}

auto ModelImpl::loadFromSharedLibrary(const std::filesystem::path& modelPath)
    -> STATUS {
    m_libModelHandle = dlopen(modelPath.string().data(), RTLD_NOW | RTLD_LOCAL);

    if (nullptr == m_libModelHandle) {
        return STATUS::FAIL;
    }

    m_composeGraphsFnHandle = reinterpret_cast<ComposeGraphsFnHandleTypeT>(
        dlsym(m_libModelHandle, "QnnModel_composeGraphs"));
    if (nullptr == m_composeGraphsFnHandle) {
        return STATUS::FAIL;
    }

    m_freeGraphInfoFnHandle = reinterpret_cast<FreeGraphInfoFnHandleTypeT>(
        dlsym(m_libModelHandle, "QnnModel_freeGraphsInfo"));
    if (nullptr == m_freeGraphInfoFnHandle) {
        return STATUS::FAIL;
    }

    return STATUS::SUCCESS;
}

}  // namespace edge::qnn
