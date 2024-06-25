<h1 align="center">
    <a href="https://runlocal.ai"><img src="./images/large-logo.png" width="300"></a>
    <br><br>
    <span style="font-size: larger;">Edgerunner</span>
    <br>
</h1>

<h4 align="center">
    A cross-platform, cross-runtime on-device AI inference library for mobile devices.
</h4>

<div align="center">
    <a href="https://runlocal.ai">Website</a> |
    <a href="https://runlocal.ai#contact">Contact</a> |
    <a href="https://discord.gg/y9EzZEkwbR">Discord</a> |
    <a href="https://x.com/Neuralize_AI">Twitter</a>
</div>

## :bulb: Introduction

The purpose of Edgerunner is to facilitate quick and easy integration of
arbitrary AI models that are targeted for consumer mobile devices
(smartphones, laptops, wearables, etc.).

Currently, on-device inference tooling is highly fragmented. To run AI models
efficiently on end-user devices, developers require a deep understanding of
every chip/device platform (Apple, Qualcomm, MediaTek, etc.) and on-device AI
frameworks (
[TFLite](https://ai.google.dev/edge/lite),
[CoreML](https://developer.apple.com/documentation/coreml),
[QNN](https://www.qualcomm.com/developer/software/neural-processing-sdk-for-ai),
etc.).
Developers also need to maintain entirely separate ML stacks for iOS and
Android, increasing the effort required to support and maintain their
cross-platform systems.

Edgerunner aims to consolidate existing on-device AI frameworks, abstracting
away cross-platform on-device AI complexities. This will be achieved through a
runtime-agnostic API, which can load and interact with arbitrary instances of
on-device AI models with just a few lines of code.

This repo is in its early stages of development (see [Features](#gift-features) and [Support](#electricplug-support)).
There is lots of work to be done in order to achieve this vision, and your
[contributions](#trophy-contributing) will be important to make this happen!

## :gift: Features

|           Feature                   |          Status          |
| ------------------------------------|:------------------------:|
| Runtime-agnostic API                | :white_check_mark:       |
| Model loading                       | :white_check_mark:       |
| Model execution                     | :white_check_mark:       |
| Automatic framework detection       | :hourglass_flowing_sand: |
| Choose optimal execution at runtime | :hourglass_flowing_sand: |
| Java bindings                       | :hourglass_flowing_sand: |
| Objective-C bindings                | :hourglass_flowing_sand: |

Please request additional features through Github issues or on our [Discord](https://discord.gg/y9EzZEkwbR).

## :electric_plug: Support

### OS

| Linux              | MacOS              | Windows                  | Android                  | iOS                      |
|:------------------:|:------------------:|:------------------------:|:------------------------:|:------------------------:|
| :white_check_mark: | :white_check_mark: | :hourglass_flowing_sand: | :hourglass_flowing_sand: | :hourglass_flowing_sand: |

### Runtime Framework

| TFLite             | CoreML                   | Onnx                     | QNN                      | OpenVino                 | Ryzen AI                 | NeuroPilot               |
|:------------------:|:------------------------:|:------------------------:|:------------------------:|:------------------------:|:------------------------:|:------------------------:|
| :white_check_mark: | :hourglass_flowing_sand: | :hourglass_flowing_sand: | :hourglass_flowing_sand: | :hourglass_flowing_sand: | :hourglass_flowing_sand: | :hourglass_flowing_sand: |

### Chip Vendor

|     | Apple                    | Qualcomm                 | MediaTek                 | Samsung                  | Intel                    | AMD                      | NVIDIA                   |
|:---:|:------------------------:|:------------------------:|:------------------------:|:------------------------:|:------------------------:|:------------------------:|:------------------------:|
| CPU | :white_check_mark:       | :white_check_mark:       | :white_check_mark:       | :white_check_mark:       | :white_check_mark:       | :white_check_mark:       | :no_entry:               |
| GPU | :hourglass_flowing_sand: | :hourglass_flowing_sand: | :hourglass_flowing_sand: | :hourglass_flowing_sand: | :white_check_mark:       | :white_check_mark:       | :hourglass_flowing_sand: |
| NPU | :hourglass_flowing_sand: | :hourglass_flowing_sand: | :hourglass_flowing_sand: | :no_entry:               | :hourglass_flowing_sand: | :hourglass_flowing_sand: | :no_entry:               |

## :hammer_and_wrench: Building and installing

See the [BUILDING](BUILDING.md) document.

## :joystick: Usage

See [examples](./example) for basic usage instructions.

## :trophy: Contributing

See the [CONTRIBUTING](CONTRIBUTING.md) document.

Join our [Discord](https://discord.gg/y9EzZEkwbR) for discussing any issues.

## :scroll: Licensing

See the [LICENSING](LICENSE.txt) document.
