#include <string>

#include <catch2/catch_test_macros.hpp>

#include "edgerunner/edgerunner.hpp"
#include "edgerunner/model.hpp"
#include "utils.hpp"

TEST_CASE("Tflite delegate ", "[tflite][delegate]") {
    const std::string modelPath = "models/tflite/mobilenet_v3_small.tflite";

    auto model = edge::createModel(modelPath);
    REQUIRE(model != nullptr);
    REQUIRE(std::string {"mobilenet_v3_small"} == model->name());
    REQUIRE(model->getDelegate() == edge::DELEGATE::CPU);

    auto delegateStatus = model->applyDelegate(edge::DELEGATE::GPU);

#ifdef EDGERUNNER_GPU
    REQUIRE(delegateStatus == edge::STATUS::SUCCESS);
    REQUIRE(model->getDelegate() == edge::DELEGATE::GPU);
#else
    REQUIRE(delegateStatus == edge::STATUS::FAIL);
    REQUIRE(model->getDelegate() == edge::DELEGATE::CPU);
#endif

    model->applyDelegate(edge::DELEGATE::CPU);
    REQUIRE(model->getDelegate() == edge::DELEGATE::CPU);

    delegateStatus = model->applyDelegate(edge::DELEGATE::NPU);

#ifdef EDGERUNNER_QNN
    REQUIRE(delegateStatus == edge::STATUS::SUCCESS);
    REQUIRE(model->getDelegate() == edge::DELEGATE::NPU);
#else
    REQUIRE(delegateStatus == edge::STATUS::FAIL);
    REQUIRE(model->getDelegate() == edge::DELEGATE::CPU);
#endif
}
