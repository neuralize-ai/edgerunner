cmake_minimum_required(VERSION 3.14)

project(edgerunnerExamples CXX)

include(../cmake/project-is-top-level.cmake)
include(../cmake/folders.cmake)

if(PROJECT_IS_TOP_LEVEL)
    find_package(edgerunner REQUIRED)
endif()

add_custom_target(run-examples)

function(add_example NAME)
    add_executable("${NAME}" "${NAME}.cpp")
    target_link_libraries("${NAME}" PRIVATE edgerunner::edgerunner)
    target_compile_features("${NAME}" PRIVATE cxx_std_17)
    add_custom_target("run_${NAME}" COMMAND "${NAME}" VERBATIM)
    add_dependencies("run_${NAME}" "${NAME}")
    add_dependencies(run-examples "run_${NAME}")
endfunction()

set(MODELS_DIR "${CMAKE_SOURCE_DIR}/models")
set(MODELS_DEST_DIR "${CMAKE_CURRENT_BINARY_DIR}/models")
set(IMAGES_DIR "${CMAKE_SOURCE_DIR}/images")
set(IMAGES_DEST_DIR "${CMAKE_CURRENT_BINARY_DIR}/images")
if(UNIX)
    execute_process(COMMAND ln -sfn ${MODELS_DIR} ${MODELS_DEST_DIR})
    execute_process(COMMAND ln -sfn ${IMAGES_DIR} ${IMAGES_DEST_DIR})
elseif(WIN32)
    execute_process(COMMAND cmd.exe /c mklink ${MODELS_DEST_DIR} ${MODELS_DIR})
    execute_process(COMMAND cmd.exe /c mklink ${IMAGES_DEST_DIR} ${IMAGES_DIR})
endif()

add_example(mobilenet_v3_small)

add_folders(Example)
