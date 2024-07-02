#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <iterator>
#include <string>
#include <vector>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>

#include "edgerunner/edgerunner.hpp"
#include "edgerunner/model.hpp"
#include "edgerunner/tensor.hpp"

TEST_CASE("Tflite from buffer default runtime (CPU)", "[tflite][buffer][cpu]") {
    const std::string badModelPath = "models/tflite/imagenet_labels.txt";
    std::ifstream badFile(badModelPath, std::ios::binary);
    std::vector<uint8_t> badModelBuffer(
        (std::istreambuf_iterator<char>(badFile)),
        std::istreambuf_iterator<char>());

    auto badModel = edge::createModel(badModelBuffer, "txt");
    REQUIRE(badModel == nullptr);

    const std::string modelPath = "models/tflite/mobilenet_v3_small.tflite";
    std::ifstream file(modelPath, std::ios::binary);
    std::vector<uint8_t> modelBuffer((std::istreambuf_iterator<char>(file)),
                                     std::istreambuf_iterator<char>());

    auto model = edge::createModel(modelBuffer, "tflite");
    REQUIRE(model != nullptr);
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

    auto outputBuffer = output->getTensorAs<float>();
    REQUIRE(outputBuffer.size() == output->getSize());

    const auto executionStatus = model->execute();
    REQUIRE(executionStatus == edge::STATUS::SUCCESS);

    BENCHMARK("execution") {
        return model->execute();
    };

    /* verify output buffer is persistent across execution */
    const auto newOutputBuffer = model->getOutput(0)->getTensorAs<float>();
    REQUIRE(outputBuffer.data() == newOutputBuffer.data());
    REQUIRE(outputBuffer.size() == newOutputBuffer.size());
}
