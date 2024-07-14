#pragma once

#include <vector>

namespace edge::qnn {

/**
 * @brief A class template for managing QNN backend configurations
 *
 * This class template allows for the management of configurations and custom
 * configurations. It provides methods for creating new configurations and
 * custom configurations, as well as retrieving pointers to QNN API compatible
 * configurations arrays.
 *
 * @tparam ConfigType The type of the configuration
 * @tparam CustomConfigType The type of the custom configuration
 */
template<typename ConfigType, typename CustomConfigType>
class Config {
  public:
    /**
     * @brief Constructor for Config class
     *
     * Initializes the default configuration and default custom configuration.
     *
     * @param defaultConfig The default configuration
     * @param defaultCustomConfig The default custom configuration
     */
    Config(ConfigType defaultConfig, CustomConfigType defaultCustomConfig)
        : m_defaultConfig(defaultConfig)
        , m_defaultCustomConfig(defaultCustomConfig) {}

    /**
     * @brief Creates a new configuration
     *
     * Creates a new configuration using the default configuration and adds it
     * to the list of configurations.
     *
     * @return A reference to the newly created configuration
     */
    auto createConfig() -> auto& {
        m_configs.push_back(m_defaultConfig);
        return m_configs.back();
    }

    /**
     * @brief Creates a new custom configuration
     *
     * Creates a new custom configuration using the default custom configuration
     * and adds it to the list of custom configurations. The returned custom
     * configuration needs to be assigned to a corresponding configuration.
     *
     * @return A reference to the newly created custom configuration
     */
    auto createCustomConfig() -> auto& {
        m_customConfigs.push_back(m_defaultCustomConfig);
        return m_customConfigs.back();
    }

    /**
     * @brief Retrieves pointers to the configurations
     *
     * Retrieves a null terminated array of pointers to all the configurations.
     *
     * @return An array of pointers to the configurations
     */
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
    ConfigType m_defaultConfig; /**< The default configuration */
    CustomConfigType
        m_defaultCustomConfig; /**< The default custom configuration */
    std::vector<ConfigType> m_configs; /**< List of configurations */
    std::vector<CustomConfigType>
        m_customConfigs; /**< List of custom configurations */
    std::vector<const ConfigType*>
        m_configPtrs; /**< List of pointers to configurations */
};

}  // namespace edge::qnn
