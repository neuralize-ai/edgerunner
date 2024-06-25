<h1 align="center">
    <br>
    <a href="https://runlocal.ai"><img src="./images/large-logo.png" width="400"></a>
    <br><br>
    <span style="font-size: larger;">Edgerunner</span>
    <br>
</h1>

<h4 align="center">
    A cross-platform, cross-runtime on-device AI inference library for mobile devices.
</h4>

<div align="center">
    <a href="https://runlocal.ai">Website</a> |
    <a href="https://discord.gg/y9EzZEkwbR">Discord</a> |
    <a href="https://x.com/Neuralize_AI">Twitter</a> |
    <a href="https://runlocal.ai#contact">Contact</a>
</div>

## Introduction

The purpose of Edgerunner is to facilitate quick and easy integration of arbitrary AI models that are targeted for consumer mobile devices (smartphones, laptops, wearables, etc.).

Currently, on-device inference tooling is highly fragmented. To run AI models efficiently on end-user devices, developers require a deep understanding of every chip/device platform (Apple, Qualcomm, MediaTek, etc.) and on-device AI frameworks ([TFLite](https://ai.google.dev/edge/lite), [CoreML](https://developer.apple.com/documentation/coreml), [QNN](https://www.qualcomm.com/developer/software/neural-processing-sdk-for-ai), etc.). Developers also need to maintain entirely separate ML stacks for iOS and Android, increasing the effort required to support and maintain their cross-platform systems.

Edgerunner aims to consolidate existing on-device AI frameworks, abstracting away cross-platform on-device AI complexities. This will be achieved through a runtime-agnostic API, which can load and interact with arbitrary instances of on-device AI models with just a few lines of code.

This repo is in its early stages of development (see [Features](#features) and [Roadmap](#roadmap)). There is lots of work to be done in order to achieve this vision, and your [contributions](#contributing) will be important to make this happen!

## Features

- Runtime-agnostic API
- Linux support
- TFLite support (XNNPACK, GPU)

### Roadmap

- Android CPU support
- Android GPU support
- Apple silicon CPU support
- Apple silicon GPU support
- Qualcomm NPU support
- MediaTek NPU support
- Samsung NPU support
- Automatic runtime detection
- Java bindings (for Android app support)
- Objective-C bindings (for Apple app support)
- Sister project implementing common pre and post-processing functionality

## Building and installing

See the [BUILDING](BUILDING.md) document.

## Contributing

See the [CONTRIBUTING](CONTRIBUTING.md) document.
Join our [Discord](https://discord.gg/y9EzZEkwbR) for feedback and discussion

## Licensing

See the [LICENSING](LICENSE.txt) document.
