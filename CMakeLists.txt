cmake_minimum_required(VERSION 3.14)

include(cmake/prelude.cmake)

project(
    edgerunner
    VERSION 0.1.0
    DESCRIPTION "Edgerunner is a cross-platform ML inference library for mobile devices"
    HOMEPAGE_URL "https://runlocal.ai"
    LANGUAGES CXX
)

include(cmake/project-is-top-level.cmake)
include(cmake/variables.cmake)

# ---- Declare library ----

add_library(
    edgerunner_edgerunner
    source/tflite/model.cpp
    source/tflite/tensor.cpp
)
add_library(edgerunner::edgerunner ALIAS edgerunner_edgerunner)

include(GenerateExportHeader)
generate_export_header(
    edgerunner_edgerunner
    BASE_NAME edgerunner
    EXPORT_FILE_NAME export/edgerunner/edgerunner_export.hpp
    CUSTOM_CONTENT_FROM_VARIABLE pragma_suppress_c4251
)

if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(
        edgerunner_edgerunner
        PUBLIC
        EDGERUNNER_STATIC_DEFINE
    )
endif()

set_target_properties(
    edgerunner_edgerunner PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    VERSION "${PROJECT_VERSION}"
    SOVERSION "${PROJECT_VERSION_MAJOR}"
    EXPORT_NAME edgerunner
    OUTPUT_NAME edgerunner
)

target_include_directories(
    edgerunner_edgerunner ${warning_guard}
    PUBLIC
    "\$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
)

target_include_directories(
    edgerunner_edgerunner SYSTEM
    PUBLIC
    "\$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/export>"
)

target_compile_features(edgerunner_edgerunner PUBLIC cxx_std_17)

find_package(fmt REQUIRED)
target_link_libraries(edgerunner_edgerunner PRIVATE fmt::fmt)

find_package(span-lite REQUIRED)
target_link_libraries(edgerunner_edgerunner PUBLIC nonstd::span-lite)

find_package(tensorflowlite REQUIRED)
target_link_libraries(edgerunner_edgerunner PUBLIC tensorflow::tensorflowlite)

if(edgerunner_ENABLE_GPU)
    target_compile_definitions(
        edgerunner_edgerunner
        PUBLIC
        EDGERUNNER_GPU
    )
endif()

# ---- Install rules ----

if(NOT CMAKE_SKIP_INSTALL_RULES)
    include(cmake/install-rules.cmake)
endif()

# ---- Developer mode ----

if(NOT edgerunner_DEVELOPER_MODE)
    return()
elseif(NOT PROJECT_IS_TOP_LEVEL)
    message(
      AUTHOR_WARNING
      "Developer mode is intended for developers of edgerunner"
  )
endif()

include(cmake/dev-mode.cmake)
