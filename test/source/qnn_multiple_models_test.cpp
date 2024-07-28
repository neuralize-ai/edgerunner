#include <string>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include "edgerunner/edgerunner.hpp"
#include "edgerunner/model.hpp"
#include "utils.hpp"

TEST_CASE("QNN multiple models", "[qnn][multiple]") {
    const std::string modelPath1 = "models/qnn/mobilenet_v3_small.so";
    const std::string modelPath2 = "models/qnn/mobilenet_v3_small.so";

    auto model1 = edge::createModel(modelPath1);
    REQUIRE(model1 != nullptr);

    auto model2 = edge::createModel(modelPath2);
    REQUIRE(model2 != nullptr);

    auto executionStatus = model1->execute();
    REQUIRE(executionStatus == edge::STATUS::SUCCESS);

    executionStatus = model2->execute();
    REQUIRE(executionStatus == edge::STATUS::SUCCESS);
}
