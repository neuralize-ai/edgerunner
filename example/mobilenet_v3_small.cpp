#include <chrono>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <vector>

#include <fmt/color.h>

#include "edgerunner/model.hpp"
#include "imageClassifier.hpp"

auto main() -> int {
    const std::filesystem::path modelPath {
        "models/tflite/mobilenet_v3_small.tflite"};
    const std::filesystem::path labelListPath {
        "models/tflite/imagenet_labels.txt"};

    ImageClassifier imageClassifier(modelPath, labelListPath);

    const size_t numPredictions = 5;

    const std::vector<std::filesystem::path> imagePaths = {
        "images/keyboard.jpg",
        "images/dog.jpg",
    };

    for (const auto& imagePath : imagePaths) {
        try {
            if (imageClassifier.loadImage(imagePath) != edge::STATUS::SUCCESS) {
                continue;
            }

            const auto start = std::chrono::high_resolution_clock::now();
            const auto predictions = imageClassifier.predict(numPredictions);
            const auto end = std::chrono::high_resolution_clock::now();
            const auto predictionTime =
                std::chrono::duration_cast<std::chrono::milliseconds>(end
                                                                      - start)
                    .count();

            fmt::print(fmt::fg(fmt::color::green), "predictions:\n");
            for (const auto& prediction : predictions) {
                fmt::print(fmt::fg(fmt::color::green),
                           "\t{} ({:.2f}%)\n",
                           prediction.first,
                           100.0F * prediction.second);
            }
            fmt::print(fmt::fg(fmt::color::yellow),
                       "prediction time: {}ms\n",
                       predictionTime);
        } catch (std::exception& ex) {
            fmt::print(fmt::fg(fmt::color::red),
                       "{} example failed: {}\n",
                       imagePath.stem().string(),
                       ex.what());
        }
    }

    return 0;
}
