#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>

#include "edgerunner/model.hpp"
#include "edgerunner/tensor.hpp"
#include "edgerunner/tflite/model.hpp"
#include "utils.hpp"

TEST_CASE("Tflite default runtime (CPU)", "[tflite][cpu]") {
    const std::string modelPath = "models/tflite/mobilenet_v3_small.tflite";
    auto model = edge::tflite::ModelImpl {modelPath};

    REQUIRE(std::string {"mobilenet_v3_small"} == model.name());

    REQUIRE(model.getDelegate() == edge::DELEGATE::CPU);

    const auto numInputs = model.getNumInputs();

    REQUIRE(numInputs == 1);

    const auto numOutputs = model.getNumOutputs();

    REQUIRE(numOutputs == 1);

    auto input = model.getInput(0);

    REQUIRE(input->getDimensions() == std::vector<size_t> {1, 224, 224, 3});

    REQUIRE(input->getType() == edge::TensorType::FLOAT32);

    auto inputData = input->getTensorAs<float>();

    REQUIRE(inputData.size() == input->getSize());

    auto output = model.getOutput(0);

    REQUIRE(output->getDimensions() == std::vector<size_t> {1, 1000});

    REQUIRE(output->getType() == edge::TensorType::FLOAT32);

    auto outputData = output->getTensorAs<float>();

    REQUIRE(outputData.size() == output->getSize());

    const auto executionStatus = model.execute();

    REQUIRE(executionStatus == edge::STATUS::SUCCESS);

    BENCHMARK("execution") {
        return model.execute();
    };

    /* verify output buffer is persistent across execution */
    const auto outputDataAfter = model.getOutput(0)->getTensorAs<float>();

    const auto outputMse = meanSquaredError(outputData, outputDataAfter);

    CAPTURE(outputMse);
    REQUIRE(outputMse < std::numeric_limits<float>::epsilon());
}
