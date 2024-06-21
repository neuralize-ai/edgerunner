cmake_minimum_required(VERSION 3.14)

project(edgerunnerTests LANGUAGES CXX)

include(../cmake/project-is-top-level.cmake)
include(../cmake/folders.cmake)

# ---- Dependencies ----

if(PROJECT_IS_TOP_LEVEL)
    find_package(edgerunner REQUIRED)
    enable_testing()
endif()

find_package(Catch2 REQUIRED)
include(Catch)

# ---- Test data ----

set(MODELS_DIR "${CMAKE_SOURCE_DIR}/models")
set(MODELS_DEST_DIR "${CMAKE_CURRENT_BINARY_DIR}/models")
if(UNIX)
    execute_process(COMMAND ln -s ${MODELS_DIR} ${MODELS_DEST_DIR})
elseif(WIN32)
    execute_process(COMMAND cmd.exe /c mklink ${MODELS_DEST_DIR} ${MODELS_DIR})
endif()

# ---- Tests ----

add_executable(edgerunner_test source/tflite_test.cpp)
target_link_libraries(
    edgerunner_test PRIVATE
    edgerunner::edgerunner
    Catch2::Catch2WithMain
)
target_compile_features(edgerunner_test PRIVATE cxx_std_17)

catch_discover_tests(edgerunner_test)

# ---- End-of-file commands ----

add_folders(Test)
