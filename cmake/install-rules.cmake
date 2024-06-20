if(PROJECT_IS_TOP_LEVEL)
  set(
      CMAKE_INSTALL_INCLUDEDIR "include/edgerunner-${PROJECT_VERSION}"
      CACHE STRING ""
  )
  set_property(CACHE CMAKE_INSTALL_INCLUDEDIR PROPERTY TYPE PATH)
endif()

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# find_package(<package>) call for consumers to find this project
set(package edgerunner)

install(
    DIRECTORY
    include/
    "${PROJECT_BINARY_DIR}/export/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    COMPONENT edgerunner_Development
)

install(
    TARGETS edgerunner_edgerunner
    EXPORT edgerunnerTargets
    RUNTIME #
    COMPONENT edgerunner_Runtime
    LIBRARY #
    COMPONENT edgerunner_Runtime
    NAMELINK_COMPONENT edgerunner_Development
    ARCHIVE #
    COMPONENT edgerunner_Development
    INCLUDES #
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

write_basic_package_version_file(
    "${package}ConfigVersion.cmake"
    COMPATIBILITY SameMajorVersion
)

# Allow package maintainers to freely override the path for the configs
set(
    edgerunner_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/${package}"
    CACHE STRING "CMake package config location relative to the install prefix"
)
set_property(CACHE edgerunner_INSTALL_CMAKEDIR PROPERTY TYPE PATH)
mark_as_advanced(edgerunner_INSTALL_CMAKEDIR)

install(
    FILES cmake/install-config.cmake
    DESTINATION "${edgerunner_INSTALL_CMAKEDIR}"
    RENAME "${package}Config.cmake"
    COMPONENT edgerunner_Development
)

install(
    FILES "${PROJECT_BINARY_DIR}/${package}ConfigVersion.cmake"
    DESTINATION "${edgerunner_INSTALL_CMAKEDIR}"
    COMPONENT edgerunner_Development
)

install(
    EXPORT edgerunnerTargets
    NAMESPACE edgerunner::
    DESTINATION "${edgerunner_INSTALL_CMAKEDIR}"
    COMPONENT edgerunner_Development
)

if(PROJECT_IS_TOP_LEVEL)
  include(CPack)
endif()
