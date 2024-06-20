#include <string>

#include "edgerunner/edgerunner.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Name is edgerunner", "[library]")
{
    auto const model = edge::Model {};
    REQUIRE(std::string("edgerunner") == model.name());
}
