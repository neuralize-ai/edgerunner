#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <iterator>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "edgerunner/edgerunner.hpp"
#include "utils.hpp"

TEST_CASE("Tflite bad model", "[tflite][misuse]") {
    const std::string badPath = "test.bin";
    auto badPathModel = edge::createModel(badPath);
    REQUIRE(badPathModel == nullptr);

    const std::string wrongFormatModelPath =
        "models/tflite/imagenet_labels.txt";
    std::ifstream wrongFormatFile(wrongFormatModelPath, std::ios::binary);
    std::vector<uint8_t> wrongFormatModelBuffer(
        (std::istreambuf_iterator<char>(wrongFormatFile)),
        std::istreambuf_iterator<char>());

    auto wrongFormatModel = edge::createModel(wrongFormatModelBuffer, "txt");
    REQUIRE(wrongFormatModel == nullptr);

    constexpr size_t ModelBufferSize {10};
    std::vector<uint8_t> badModelBuffer(ModelBufferSize);

    auto badBufferModel = edge::createModel(badModelBuffer);
    REQUIRE(badBufferModel == nullptr);

    const std::filesystem::path badModelPath {"badModel.tflite"};
    std::ofstream badModelFile(badModelPath, std::ios::binary);
    for (const auto& element : badModelPath) {
        badModelFile << element;
    }

    auto badFileModel = edge::createModel(badModelPath);
    REQUIRE(badFileModel == nullptr);
}
