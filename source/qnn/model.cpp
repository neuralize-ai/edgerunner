#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <ios>
#include <memory>
#include <string>
#include <vector>

#include "edgerunner/model.hpp"

#include <nonstd/span.hpp>

#include "edgerunner/qnn/backend.hpp"
#include "edgerunner/qnn/model.hpp"
#include "edgerunner/qnn/tensor.hpp"
#include "edgerunner/tensor.hpp"

namespace edge::qnn {

std::unique_ptr<Backend> ModelImpl::m_backend = nullptr;

ModelImpl::ModelImpl(const std::filesystem::path& modelPath)
    : Model(modelPath) {
    const auto modelExtension = modelPath.extension().string().substr(1);
    m_loadCachedBinary = modelExtension == "bin";

    initializeBackend();

    if (!m_loadCachedBinary) {
        setCreationStatus(loadModel(modelPath));
        setCreationStatus(composeGraphs());
        setPrecision(detectPrecision());
        setCreationStatus(
            m_graph.setGraphConfig(m_backend->getDelegate(), getPrecision()));
        setCreationStatus(m_graph.finalizeGraphs());

        // m_graphInfo.saveContextBinary(name() + ".bin");
    } else {
        setCreationStatus(m_graph.loadSystemLibrary());

        std::ifstream file(modelPath, std::ios::binary);
        if (!file) {
            setCreationStatus(STATUS::FAIL);
            return;
        }

        const auto bufferSize = std::filesystem::file_size(modelPath);

        std::vector<uint8_t> modelBuffer(bufferSize);

        if (!file.read(
                reinterpret_cast<char*> /* NOLINT */ (modelBuffer.data()),
                static_cast<std::streamsize>(modelBuffer.size())))
        {
            setCreationStatus(STATUS::FAIL);
            return;
        }

        setCreationStatus(loadModel(modelBuffer));
    }

    setCreationStatus(allocate());
}

ModelImpl::ModelImpl(const nonstd::span<uint8_t>& modelBuffer) {
    setCreationStatus(loadModel(modelBuffer));
}

auto ModelImpl::loadModel(const std::filesystem::path& modelPath) -> STATUS {
    return m_graph.loadFromSharedLibrary(modelPath);
}

auto ModelImpl::loadModel(const nonstd::span<uint8_t>& modelBuffer) -> STATUS {
    return loadFromContextBinary(modelBuffer);
}

auto ModelImpl::applyDelegate(const DELEGATE& delegate) -> STATUS {
    if (delegate != DELEGATE::NPU) {
        return STATUS::FAIL;
    }

    setDelegate(delegate);

    return STATUS::SUCCESS;
}

auto ModelImpl::execute() -> STATUS {
    return m_graph.execute();
}

auto ModelImpl::loadFromContextBinary(const nonstd::span<uint8_t>& modelBuffer)
    -> STATUS {
    auto& qnnInterface = m_backend->getInterface();
    auto& backendHandle = m_backend->getHandle();
    auto& deviceHandle = m_backend->getDeviceHandle();

    if (m_graph.loadContextFromBinary(
            qnnInterface, backendHandle, deviceHandle, modelBuffer)
        != STATUS::SUCCESS)
    {
        return STATUS::FAIL;
    }

    return m_graph.retrieveGraphFromContext();
}

auto ModelImpl::composeGraphs() -> STATUS {
    auto& qnnInterface = m_backend->getInterface();
    auto& qnnBackendHandle = m_backend->getHandle();
    auto& qnnDeviceHandle = m_backend->getDeviceHandle();

    m_graph.createContext(qnnInterface, qnnBackendHandle, qnnDeviceHandle);

    return m_graph.composeGraphs(qnnBackendHandle);
}

auto ModelImpl::detectPrecision() -> TensorType {
    const auto inputTensorSpecs = m_graph.getInputs();

    std::vector<TensorImpl> inputs;
    inputs.reserve(inputTensorSpecs.size());
    for (auto& inputTensorSpec : inputTensorSpecs) {
        inputs.emplace_back(&inputTensorSpec, false);
    }

    for (auto& input : inputs) {
        const auto type = input.getType();

        if (type == TensorType::FLOAT16 || type == TensorType::FLOAT32) {
            return TensorType::FLOAT16;
        }
    }

    return TensorType::UINT8;
}

auto ModelImpl::allocate() -> STATUS {
    auto& inputs = getInputs();
    auto& outputs = getOutputs();

    inputs.clear();
    outputs.clear();

    const auto inputTensorSpecs = m_graph.getInputs();
    const auto outputTensorSpecs = m_graph.getOutputs();

    if (inputTensorSpecs.data() == nullptr
        || outputTensorSpecs.data() == nullptr)
    {
        return STATUS::FAIL;
    }

    inputs.reserve(inputTensorSpecs.size());
    for (auto& inputTensorSpec : inputTensorSpecs) {
        inputs.emplace_back(std::make_shared<TensorImpl>(&inputTensorSpec));
    }

    outputs.reserve(outputTensorSpecs.size());
    for (auto& outputTensorSpec : outputTensorSpecs) {
        outputs.emplace_back(std::make_shared<TensorImpl>(&outputTensorSpec));
    }

    return STATUS::SUCCESS;
}

}  // namespace edge::qnn
