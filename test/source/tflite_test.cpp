#include <string>

#include <catch2/catch_test_macros.hpp>
#include <fmt/core.h>
#include <fmt/ranges.h>

#include "edgerunner/tflite/model.hpp"

TEST_CASE("Name is edgerunner", "[library]") {
    const std::string modelPath = "models/tflite/mobilenet_v3_small.tflite";
    auto model = edge::tflite::ModelImpl {modelPath};

    REQUIRE(std::string {"mobilenet_v3_small"} == model.name());

    const auto numInputs = model.getNumInputs();

    REQUIRE(numInputs == 1);

    const auto numOutputs = model.getNumOutputs();

    REQUIRE(numOutputs == 1);

    auto input = model.getInput(0);

    fmt::print("input {}:\n", input->getName());
    fmt::print("\tsize: {}\n", input->getSize());
    fmt::print("\tdimensions: {}\n", input->getDimensions());
    fmt::print("\ttype: {}\n", static_cast<int>(input->getType()));

    REQUIRE(input->getDimensions() == std::vector<size_t> {1, 224, 224, 3});

    REQUIRE(input->getType() == edge::TensorType::FLOAT32);

    auto inputData = input->getTensorAs<float>();

    REQUIRE(inputData.size() == input->getSize());

    auto output = model.getOutput(0);

    fmt::print("output {}:\n", output->getName());
    fmt::print("\tsize: {}\n", output->getSize());
    fmt::print("\tdimensions: {}\n", output->getDimensions());
    fmt::print("\ttype: {}\n", static_cast<int>(output->getType()));

    REQUIRE(output->getDimensions() == std::vector<size_t> {1, 1000});

    REQUIRE(output->getType() == edge::TensorType::FLOAT32);

    auto outputData = output->getTensorAs<float>();

    REQUIRE(outputData.size() == output->getSize());

    fmt::print("output before:\n {}\n", outputData);

    model.execute();

    fmt::print("output after:\n {}\n", outputData);

    REQUIRE(false);
}
