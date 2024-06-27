#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include "edgerunner/edgerunner.hpp"
#include "edgerunner/model.hpp"
#include "edgerunner/tensor.hpp"
#include "utils.hpp"

TEST_CASE("Tflite GPU runtime", "[tflite][gpu]") {
    const std::string modelPath = "models/tflite/mobilenet_v3_small.tflite";
    auto model = edge::createModel(modelPath);

    REQUIRE(model != nullptr);

    REQUIRE(std::string {"mobilenet_v3_small"} == model->name());

    REQUIRE(model->getDelegate() == edge::DELEGATE::CPU);

    auto cpuInputData = model->getInput(0)->getTensorAs<float>();
    for (auto& cpuInputDatum : cpuInputData) {
        cpuInputDatum = 0;
    }

    model->execute();

    const auto cpuOutput = model->getOutput(0)->getTensorAs<float>();

    /* applying a new delegate releases memory, so need to copy CPU output to
     * compare later */
    std::vector<float> cpuResult;
    cpuResult.reserve(cpuOutput.size());
    std::copy(
        cpuOutput.cbegin(), cpuOutput.cend(), std::back_inserter(cpuResult));

    const auto delegateStatus = model->applyDelegate(edge::DELEGATE::GPU);

    REQUIRE(delegateStatus == edge::STATUS::SUCCESS);

    REQUIRE(model->getDelegate() == edge::DELEGATE::GPU);

    const auto numInputs = model->getNumInputs();

    REQUIRE(numInputs == 1);

    const auto numOutputs = model->getNumOutputs();

    REQUIRE(numOutputs == 1);

    auto input = model->getInput(0);

    REQUIRE(input->getName() == "image_tensor");

    REQUIRE(input->getDimensions() == std::vector<size_t> {1, 224, 224, 3});

    REQUIRE(input->getType() == edge::TensorType::FLOAT32);

    auto inputData = input->getTensorAs<float>();

    REQUIRE(inputData.size() == input->getSize());

    for (auto& inputDatum : inputData) {
        inputDatum = 0;
    }

    const auto executionStatus = model->execute();

    REQUIRE(executionStatus == edge::STATUS::SUCCESS);

    auto output = model->getOutput(0);

    REQUIRE(output->getName() == "output_0");

    REQUIRE(output->getDimensions() == std::vector<size_t> {1, 1000});

    REQUIRE(output->getType() == edge::TensorType::FLOAT32);

    auto outputData = output->getTensorAs<float>();

    REQUIRE(outputData.size() == output->getSize());

    const auto mse = meanSquaredError(cpuResult, outputData);

    CAPTURE(mse);
    REQUIRE(mse < mseThreshold);

    BENCHMARK("execution") {
        return model->execute();
    };
}
