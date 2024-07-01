#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include "edgerunner/tflite/tensor.hpp"

TEST_CASE("Tflite bad tensor", "[tflite][cpu][tensor]") {
    edge::tflite::TensorImpl tensor;

    const auto name = tensor.getName();
    REQUIRE(name == "");

    const auto dimensions = tensor.getDimensions();
    REQUIRE(dimensions.empty());

    const auto size = tensor.getSize();
    REQUIRE(size == 0);
}
