#include <string>

#include <catch2/catch_test_macros.hpp>

#include "edgerunner/dummy/dummy.hpp"

TEST_CASE("Name is edgerunner", "[library]") {
    auto model = edge::Dummy {};
    model.loadModel("model.bin");
    REQUIRE(std::string("model") == model.name());
}
