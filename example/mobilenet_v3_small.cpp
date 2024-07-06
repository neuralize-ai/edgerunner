#include <chrono>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <ratio>
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

#ifdef EDGERUNNER_QNN
    imageClassifier.setDelegate(edge::DELEGATE::NPU);
#elif EDGERUNNER_GPU
    imageClassifier.setDelegate(edge::DELEGATE::GPU);
#endif

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
            const auto [predictions, inferenceTime] =
                imageClassifier.predict(numPredictions);
            const auto end = std::chrono::high_resolution_clock::now();
            const auto predictionTime =
                std::chrono::duration<double, std::milli>(end - start).count();

            fmt::print(stderr,
                       fmt::fg(fmt::color::green),
                       "predictions for {}:\n",
                       imagePath.filename().string());
            for (const auto& prediction : predictions) {
                fmt::print(stderr,
                           fmt::fg(fmt::color::green),
                           "\t{} ({:.2f}%)\n",
                           prediction.first,
                           100.0F * prediction.second);
            }
            fmt::print(stderr,
                       fmt::fg(fmt::color::yellow),
                       "prediction time: {}ms\n",
                       predictionTime);
            fmt::print(stderr,
                       fmt::fg(fmt::color::yellow),
                       "inference time: {}ms\n",
                       inferenceTime);
        } catch (std::exception& ex) {
            fmt::print(stderr,
                       fmt::fg(fmt::color::red),
                       "{} example failed: {}\n",
                       imagePath.stem().string(),
                       ex.what());
        }
    }

    return 0;
}
