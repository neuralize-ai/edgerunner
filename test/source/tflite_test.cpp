#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include "edgerunner/edgerunner.hpp"
#include "edgerunner/model.hpp"
#include "edgerunner/tensor.hpp"
#include "utils.hpp"

TEST_CASE("Tflite default runtime (CPU)", "[tflite][cpu]") {
    const std::string modelPath = "models/tflite/mobilenet_v3_small.tflite";
    auto model = edge::createModel(modelPath);
    REQUIRE(model != nullptr);
    REQUIRE(std::string {"mobilenet_v3_small"} == model->name());
    REQUIRE(model->getDelegate() == edge::DELEGATE::CPU);

    const auto inputs = model->getInputs();
    const auto numInputs = model->getNumInputs();
    REQUIRE(numInputs == 1);
    REQUIRE(numInputs == inputs.size());

    const auto outputs = model->getOutputs();
    const auto numOutputs = model->getNumOutputs();
    REQUIRE(numOutputs == 1);
    REQUIRE(numOutputs == outputs.size());

    auto input = model->getInput(0);
    REQUIRE(input->getName() == "image_tensor");
    REQUIRE(input->getDimensions() == std::vector<size_t> {1, 224, 224, 3});
    REQUIRE(input->getType() == edge::TensorType::FLOAT32);
    REQUIRE(input.get() == inputs[0].get());

    auto inputData = input->getTensorAs<float>();
    REQUIRE(inputData.size() == input->getSize());

    auto output = model->getOutput(0);
    REQUIRE(output->getName() == "output_0");
    REQUIRE(output->getDimensions() == std::vector<size_t> {1, 1000});
    REQUIRE(output->getType() == edge::TensorType::FLOAT32);
    REQUIRE(output.get() == outputs[0].get());

    auto outputData = output->getTensorAs<float>();
    REQUIRE(outputData.size() == output->getSize());

    const auto executionStatus = model->execute();
    REQUIRE(executionStatus == edge::STATUS::SUCCESS);

    BENCHMARK("execution") {
        return model->execute();
    };

    /* verify output buffer is persistent across execution */
    const auto outputDataAfter = model->getOutput(0)->getTensorAs<float>();
    const auto outputMse = meanSquaredError(outputData, outputDataAfter);
    CAPTURE(outputMse);
    REQUIRE(outputMse < std::numeric_limits<float>::epsilon());
}
