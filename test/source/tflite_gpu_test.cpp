#include <algorithm>
#include <cstddef>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>

#include "edgerunner/model.hpp"
#include "edgerunner/tensor.hpp"
#include "edgerunner/tflite/model.hpp"

constexpr float mseThreshold = 4.0;

template<typename C1, typename C2, typename T = typename C1::value_type>
auto meanSquaredError(const C1& input1, const C2& input2) -> T {
    return std::transform_reduce(input1.cbegin(),
                                 input1.cend(),
                                 input2.cbegin(),
                                 static_cast<T>(0),
                                 std::plus<>(),
                                 [](auto val1, auto val2) {
                                     const auto error = val1 - val2;
                                     return error * error;
                                 })
        / static_cast<T>(input1.size());
}

TEST_CASE("Tflite GPU runtime", "[tflite][gpu]") {
    const std::string modelPath = "models/tflite/mobilenet_v3_small.tflite";
    auto model = edge::tflite::ModelImpl {modelPath};

    REQUIRE(std::string {"mobilenet_v3_small"} == model.name());

    model.execute();

    const auto cpuOutput = model.getOutput(0)->getTensorAs<float>();

    std::vector<float> cpuResult(cpuOutput.size());
    std::copy(cpuOutput.cbegin(), cpuOutput.cend(), cpuResult.begin());

    model.setDelegate(edge::DELEGATE::GPU);
    const auto delegateStatus = model.applyDelegate();

    REQUIRE(delegateStatus == edge::STATUS::SUCCESS);

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

    const auto mse = meanSquaredError(cpuResult, outputData);

    REQUIRE(mse < mseThreshold);

    BENCHMARK("execution") {
        return model.execute();
    };
}
