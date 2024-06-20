#include "edgerunner/edgerunner.hpp"

namespace edge {

class Dummy : public Model<std::string> {
  public:
    void loadModel(const std::filesystem::path& modelPath) final {
        m_modelPath = modelPath;
        setName(modelPath.stem());
    }

    void execute() final {
        std::string result;
        for (const auto& input : accessInputs()) {
            result += input;
        }

        for (auto& output : accessOutputs()) {
            output = result;
        }
    }

  private:
    std::filesystem::path m_modelPath;
};

}  // namespace edge
