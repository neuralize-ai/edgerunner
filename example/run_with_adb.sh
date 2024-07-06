#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/.."

usage="Usage: $(basename "$0") [-h,--help] [-e,--example <example name>] [-b,--build-dir <build directory>] -- script to execute edgerunner examples on Android devices"

EXAMPLE=""
BUILD_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
    -h | --help)
        echo "$usage"
        exit
        ;;
    -e | --example)
        shift
        if [[ -z "$1" || "$1" == --* ]]; then
            echo "Error: -e,--example requires an argument."
            echo "$usage"
            exit 1
        fi
        EXAMPLE="$1"
        shift
        ;;
    -b | --build-dir)
        shift
        if [[ -z "$1" || "$1" == --* ]]; then
            echo "Error: -b,--build-dir requires an argument."
            echo "$usage"
            exit 1
        fi
        BUILD_DIR="$1"
        shift
        ;;
    *)
        echo "Unsupported option: $1"
        exit 1
        ;;
    esac
done

if [[ -z "$EXAMPLE" ]]; then
    echo "Error: -e,--example is a mandatory argument."
    echo "$usage"
    exit 1
fi

if [[ -z "$BUILD_DIR" ]]; then
    echo "Error: -b,--build-dir is a mandatory argument."
    echo "$usage"
    exit 1
fi

MODELS_DIR="${ROOT_DIR}/models"
IMAGES_DIR="${ROOT_DIR}/images"

APP_ROOT_DIR="/data/local/tmp/edgerunner"
APP_BUILD_DIR="${APP_ROOT_DIR}/build"
APP_EXAMPLE_DIR="${APP_BUILD_DIR}/prod/example"

adb shell "mkdir -p ${APP_BUILD_DIR}"

adb push --sync "${BUILD_DIR}" "${APP_BUILD_DIR}"
adb push --sync "${MODELS_DIR}" "${APP_EXAMPLE_DIR}"
adb push --sync "${IMAGES_DIR}" "${APP_EXAMPLE_DIR}"

adb shell "cd ${APP_EXAMPLE_DIR} && LD_LIBRARY_PATH=.. ADSP_LIBRARY_PATH=.. ./${EXAMPLE}"

adb shell "rm -rf ${APP_ROOT_DIR}"
