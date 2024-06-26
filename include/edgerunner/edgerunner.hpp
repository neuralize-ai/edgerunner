/**
 * @file edgerunner.hpp
 * @brief Header file for the Model Factory
 */

#pragma once

#include <filesystem>
#include <memory>

#include "edgerunner/edgerunner_export.hpp"
#include "model.hpp"

namespace edge {

/**
 * @brief Function to create a model from a given file path
 *
 * This function takes a file path to a model and creates a new Model object
 * from it.
 *
 * This is the intended way to instanciate a Model using the edgerunner library
 *
 * @param modelPath The file path to the model file
 * @return A unique pointer to the created Model object
 */
auto EDGERUNNER_EXPORT createModel(const std::filesystem::path& modelPath)
    -> std::unique_ptr<Model>;

}  // namespace edge
