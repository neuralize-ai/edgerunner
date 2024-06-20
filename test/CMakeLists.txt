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

# ---- Tests ----

add_executable(edgerunner_test source/edgerunner_test.cpp)
target_link_libraries(
    edgerunner_test PRIVATE
    edgerunner::edgerunner
    Catch2::Catch2WithMain
)
target_compile_features(edgerunner_test PRIVATE cxx_std_17)

catch_discover_tests(edgerunner_test)

# ---- End-of-file commands ----

add_folders(Test)
