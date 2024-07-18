cmake_minimum_required(VERSION 3.14)

project(edgerunnerExamples CXX)

include(../cmake/project-is-top-level.cmake)
include(../cmake/folders.cmake)

add_library(examples_interface INTERFACE)

if(PROJECT_IS_TOP_LEVEL)
    find_package(edgerunner REQUIRED)
    find_package(fmt REQUIRED)
endif()

target_link_libraries(examples_interface INTERFACE edgerunner::edgerunner)

target_link_libraries(examples_interface INTERFACE fmt::fmt)

find_package(OpenCV REQUIRED)
target_link_libraries(examples_interface INTERFACE opencv::opencv)

add_custom_target(run-examples)

function(add_example NAME)
    add_executable("${NAME}" "${NAME}.cpp")
    target_link_libraries("${NAME}" PRIVATE examples_interface)
    target_compile_features("${NAME}" PRIVATE cxx_std_17)
    if(ANDROID)
        add_custom_target(
            "run_${NAME}"
            COMMAND "${CMAKE_SOURCE_DIR}/example/run_with_adb.sh" -b
                    "${CMAKE_BINARY_DIR}" -e "${NAME}"
            VERBATIM
        )
    else()
        add_custom_target(
            "run_${NAME}"
            COMMAND "${NAME}"
            VERBATIM
        )
    endif()
    add_dependencies("run_${NAME}" "${NAME}")
    add_dependencies(run-examples "run_${NAME}")
endfunction()

# NOTE: for Android, adb push fails on symlinks, push directly manually instead
if(ANDROID)
    # file(COPY "${CMAKE_BINARY_DIR}/../runtimeLibs/" DESTINATION
    # ${CMAKE_CURRENT_BINARY_DIR} )
    foreach(dir ${CONAN_RUNTIME_LIB_DIRS})
        file(GLOB_RECURSE shared_libs "${dir}/*.so")
        file(COPY ${shared_libs} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
    endforeach()
else()
    set(MODELS_DIR "${CMAKE_SOURCE_DIR}/models")
    set(MODELS_DEST_DIR "${CMAKE_CURRENT_BINARY_DIR}/models")
    set(IMAGES_DIR "${CMAKE_SOURCE_DIR}/images")
    set(IMAGES_DEST_DIR "${CMAKE_CURRENT_BINARY_DIR}/images")
    if(UNIX)
        execute_process(COMMAND ln -sfn ${MODELS_DIR} ${MODELS_DEST_DIR})
        execute_process(COMMAND ln -sfn ${IMAGES_DIR} ${IMAGES_DEST_DIR})
    elseif(WIN32)
        execute_process(
            COMMAND cmd.exe /c mklink ${MODELS_DEST_DIR} ${MODELS_DIR}
        )
        execute_process(
            COMMAND cmd.exe /c mklink ${IMAGES_DEST_DIR} ${IMAGES_DIR}
        )
    endif()
endif()

add_example(mobilenet_v3_small)

add_folders(Example)
