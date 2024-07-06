#pragma once

#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <string>

#include <fmt/core.h>
#include <fmt/ranges.h>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>

#include "edgerunner/edgerunner.hpp"
#include "edgerunner/model.hpp"

class ImageClassifier {
  public:
    ImageClassifier(const std::filesystem::path& modelPath,
                    const std::filesystem::path& labelListPath);

    auto loadImage(const std::filesystem::path& imagePath) -> edge::STATUS;

    auto setDelegate(const edge::DELEGATE delegate) -> edge::STATUS;

    auto predict(size_t numPredictions = 3)
        -> std::pair<std::vector<std::pair<std::string, float>>, double>;

  private:
    static void toRGBFloat(cv::Mat& image);

    static void resize(cv::Mat& image, size_t size);

    static void centerCrop(cv::Mat& image, const std::vector<size_t>& cropSize);

    static void normalize(cv::Mat& image);

    static void writeImageToInputBuffer(const cv::Mat& inputImage,
                                        nonstd::span<float>& output);

    static void preprocess(cv::Mat& image,
                           const std::vector<size_t>& dimensions,
                           nonstd::span<float>& modelInput);

    static auto softmax(const nonstd::span<float>& input) -> std::vector<float>;

    static auto topKIndices(const nonstd::span<float>& elements,
                            size_t numPredictions) -> std::vector<size_t>;

    static auto loadLabelList(const std::filesystem::path& labelListPath)
        -> std::vector<std::string>;

    static void printPixel(const nonstd::span<float>& image,
                           const std::vector<size_t>& dimensions,
                           size_t hIndex,
                           size_t wIndex);

    static void printPixel(const cv::Mat& image, size_t hIndex, size_t wIndex);

    template<typename T>
    static auto nextPowerOfTwo(T val) -> T {
        T power = 2;
        while (power < val) {
            power *= 2;
        }

        return power;
    }

    std::unique_ptr<edge::Model> m_model;

    std::vector<std::string> m_labelList;

    cv::Mat m_image;
};

inline ImageClassifier::ImageClassifier(
    const std::filesystem::path& modelPath,
    const std::filesystem::path& labelListPath)
    : m_model(edge::createModel(modelPath))
    , m_labelList(loadLabelList(labelListPath)) {}

inline auto ImageClassifier::loadImage(const std::filesystem::path& imagePath)
    -> edge::STATUS {
    m_image = cv::imread(imagePath, cv::IMREAD_COLOR);

    if (m_image.empty()) {
        return edge::STATUS::FAIL;
    }

    toRGBFloat(m_image);

    return edge::STATUS::SUCCESS;
}

inline auto ImageClassifier::setDelegate(const edge::DELEGATE delegate)
    -> edge::STATUS {
    return m_model->applyDelegate(delegate);
}

inline auto ImageClassifier::predict(const size_t numPredictions)
    -> std::pair<std::vector<std::pair<std::string, float>>, double> {
    auto input = m_model->getInput(0);

    const auto inputDimensions = input->getDimensions();

    auto inputBuffer = input->getTensorAs<float>();

    preprocess(m_image, inputDimensions, inputBuffer);

    const auto start = std::chrono::high_resolution_clock::now();
    if (m_model->execute() != edge::STATUS::SUCCESS) {
        return {};
    }
    const auto end = std::chrono::high_resolution_clock::now();

    const auto inferenceTime =
        std::chrono::duration<double, std::milli>(end - start).count();

    auto output = m_model->getOutput(0)->getTensorAs<float>();

    const auto probabilities = softmax(output);

    const auto topIndices = topKIndices(output, numPredictions);

    std::vector<std::pair<std::string, float>> topPredictions;
    topPredictions.reserve(topIndices.size());

    for (const auto index : topIndices) {
        topPredictions.emplace_back(m_labelList[index + 1],
                                    probabilities[index]);
    }

    return {topPredictions, inferenceTime};
}

inline void ImageClassifier::toRGBFloat(cv::Mat& image) {
    // Convert the image to float and scale it to [0, 1] range
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    image.convertTo(image, CV_32FC3, 1.0 / std::numeric_limits<uint8_t>::max());
}

inline void ImageClassifier::resize(cv::Mat& image, const size_t size) {
    const auto imageHeight = image.rows;
    const auto imageWidth = image.cols;

    const auto longDim = static_cast<float>(std::max(imageHeight, imageWidth));
    const auto shortDim = static_cast<float>(std::min(imageHeight, imageWidth));

    const auto newLong =
        static_cast<size_t>(static_cast<float>(size) * longDim / shortDim);

    const auto newHeight =
        static_cast<int>((imageHeight > imageWidth) ? newLong : size);
    const auto newWidth =
        static_cast<int>((imageHeight > imageWidth) ? size : newLong);

    cv::resize(
        image, image, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_LINEAR);
}

inline void ImageClassifier::centerCrop(cv::Mat& image,
                                        const std::vector<size_t>& cropSize) {
    auto imageHeight = image.rows;
    auto imageWidth = image.cols;

    const auto cropHeight = static_cast<int>(cropSize[0]);
    const auto cropWidth = static_cast<int>(cropSize[1]);

    if (cropHeight > imageWidth || cropWidth > imageHeight) {
        const auto padLeft = (cropHeight - imageWidth) / 2;
        const auto padTop = (cropWidth - imageHeight) / 2;
        const auto padRight = (cropHeight - imageWidth + 1) / 2;
        const auto padBottom = (cropWidth - imageHeight + 1) / 2;

        cv::copyMakeBorder(image,
                           image,
                           padTop,
                           padBottom,
                           padLeft,
                           padRight,
                           cv::BORDER_CONSTANT,
                           cv::Scalar(0, 0, 0));
        imageHeight = image.rows;
        imageWidth = image.cols;
    }

    const auto cropTop =
        static_cast<int>(std::floor((imageHeight - cropWidth) / 2.0));
    const auto cropLeft =
        static_cast<int>(std::floor((imageWidth - cropHeight) / 2.0));

    const cv::Rect cropRegion(cropLeft, cropTop, cropHeight, cropWidth);
    image = image(cropRegion);
}

inline void ImageClassifier::normalize(cv::Mat& image) {
    const cv::Scalar mean(0.485, 0.456, 0.406);
    const cv::Scalar std(0.229, 0.224, 0.225);

    cv::subtract(image, mean, image);
    cv::divide(image, std, image);
}

inline void ImageClassifier::writeImageToInputBuffer(
    const cv::Mat& inputImage, nonstd::span<float>& output) {
    const auto height = static_cast<size_t>(inputImage.rows);
    const auto width = static_cast<size_t>(inputImage.cols);

    const auto numChannels = static_cast<size_t>(inputImage.channels());
    const auto rowSize = width * numChannels;

    for (size_t i = 0; i < height; ++i) {
        const auto hOffset = i * rowSize;
        for (size_t j = 0; j < width; ++j) {
            const auto wOffset = hOffset + j * numChannels;
            const auto& pixel = inputImage.at<cv::Vec3f>(static_cast<int>(i),
                                                         static_cast<int>(j));
            output[wOffset] = pixel[0];
            output[wOffset + 1] = pixel[1];
            output[wOffset + 2] = pixel[2];
        }
    }
}

inline void ImageClassifier::preprocess(cv::Mat& image,
                                        const std::vector<size_t>& dimensions,
                                        nonstd::span<float>& modelInput) {
    const auto resizedSize = nextPowerOfTwo(dimensions[1]);
    resize(image, resizedSize);

    const std::vector<size_t> cropDimensions = {dimensions[1], dimensions[2]};
    centerCrop(image, cropDimensions);

    normalize(image);

    writeImageToInputBuffer(image, modelInput);
}

inline auto ImageClassifier::softmax(const nonstd::span<float>& input)
    -> std::vector<float> {
    const float maxInput = *std::max_element(input.cbegin(), input.cend());

    std::vector<float> softmaxValues;
    softmaxValues.reserve(input.size());

    std::transform(input.cbegin(),
                   input.cend(),
                   std::back_inserter(softmaxValues),
                   [maxInput](auto val) { return std::exp(val - maxInput); });

    const auto expSum =
        std::accumulate(softmaxValues.begin(), softmaxValues.end(), 0.0F);

    std::transform(softmaxValues.begin(),
                   softmaxValues.end(),
                   softmaxValues.begin(),
                   [expSum](auto val) { return val / expSum; });
    return softmaxValues;
}

inline auto ImageClassifier::topKIndices(const nonstd::span<float>& elements,
                                         const size_t numPredictions)
    -> std::vector<size_t> {
    std::vector<size_t> indices(elements.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(
        indices.begin(),
        indices.begin()
            + static_cast<std::vector<size_t>::difference_type>(numPredictions),
        indices.end(),
        [&elements](size_t val1, size_t val2) {
            return elements[val1] > elements[val2];
        });
    indices.resize(numPredictions);
    return indices;
}

inline auto ImageClassifier::loadLabelList(
    const std::filesystem::path& labelListPath) -> std::vector<std::string> {
    std::vector<std::string> labels;
    std::ifstream file(labelListPath);
    std::string line;
    while (std::getline(file, line)) {
        labels.push_back(line);
    }
    return labels;
}

inline void ImageClassifier::printPixel(const nonstd::span<float>& image,
                                        const std::vector<size_t>& dimensions,
                                        size_t hIndex,
                                        size_t wIndex) {
    const auto red =
        *(image.cbegin() + hIndex * dimensions[2] * 3 + wIndex * 3);
    const auto green =
        *(image.cbegin() + hIndex * dimensions[2] * 3 + wIndex * 3 + 1);
    const auto blue =
        *(image.cbegin() + hIndex * dimensions[2] * 3 + wIndex * 3 + 2);

    fmt::print(stderr,
               "pixel ({}, {}): [{}, {}, {}]\n",
               hIndex,
               wIndex,
               red,
               green,
               blue);
}

inline void ImageClassifier::printPixel(const cv::Mat& image,
                                        size_t hIndex,
                                        size_t wIndex) {
    auto pixel =
        image.at<cv::Vec3f>(static_cast<int>(hIndex), static_cast<int>(wIndex));
    auto red = pixel[0];
    auto green = pixel[1];
    auto blue = pixel[2];

    fmt::print(stderr,
               "pixel ({}, {}): [{}, {}, {}]\n",
               hIndex,
               wIndex,
               red,
               green,
               blue);
}
