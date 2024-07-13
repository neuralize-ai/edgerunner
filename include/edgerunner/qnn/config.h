#include <vector>

namespace edge::qnn {

template<typename ConfigType, typename CustomConfigType>
class Config {
  public:
    Config(ConfigType defaultConfig, CustomConfigType defaultCustomConfig)
        : m_defaultConfig(defaultConfig)
        , m_defaultCustomConfig(defaultCustomConfig) {}

    auto createConfig() -> auto& {
        m_configs.push_back(m_defaultConfig);
        return m_configs.back();
    }

    auto createCustomConfig() -> auto& {
        m_customConfigs.push_back(m_defaultCustomConfig);
        return m_customConfigs.back();
    }

    auto getPtr() -> const ConfigType** {
        m_configPtrs.clear();
        m_configPtrs.reserve(m_configs.size() + 1);
        for (auto& config : m_configs) {
            m_configPtrs.push_back(&config);
        }
        m_configPtrs.push_back(nullptr);
        return m_configPtrs.data();
    }

  private:
    ConfigType m_defaultConfig;
    CustomConfigType m_defaultCustomConfig;
    std::vector<ConfigType> m_configs;
    std::vector<CustomConfigType> m_customConfigs;
    std::vector<const ConfigType*> m_configPtrs;
};

}  // namespace edge::qnn
