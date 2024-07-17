<h1 align="center">
    <a href="https://runlocal.ai">
        <img src="./images/large-logo.png" width="300">
    </a>
    <br><br>
    Edgerunner
</h1>

<h4 align="center">
    Simplified AI runtime integration for mobile app development
</h4>

<div align="center">
    <a href="https://runlocal.ai">Website</a> |
    <a href="https://runlocal.ai#contact">Contact</a> |
    <a href="https://discord.gg/y9EzZEkwbR">Discord</a> |
    <a href="https://x.com/Neuralize_AI">Twitter</a> |
    <a href="https://neuralize-ai.github.io/edgerunner">Docs</a>
    <br><br>
    <img src="https://github.com/neuralize-ai/edgerunner/actions/workflows/ci.yml/badge.svg"/>
    <a href="https://codecov.io/gh/neuralize-ai/edgerunner" >
        <img src="https://codecov.io/gh/neuralize-ai/edgerunner/graph/badge.svg?token=W05G243LIK"/>
    </a>
</div>

## 💡 Introduction

The purpose of Edgerunner is to facilitate easy integration of
common AI model formats and inference runtimes
([TFLite](https://ai.google.dev/edge/lite), [ONNX](https://github.com/onnx/onnx),
[QNN](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk),
etc.) for
consumer mobile devices (smartphones, laptops, tablets, wearables, etc.).

Edgerunner removes the complexities of deploying an off-the-shelf model into
your app, with NPU acceleration, regardless of the model format or target devices.
Platform-specific NPU SDKs are managed for you and can be leveraged with a
device-agnostic API. Edgerunner exposes a boilerplate-free API with sane defaults.

Kotlin bindings to create AI applications in Android with Edgerunner can
be found at
[edgerunner-android](https://github.com/neuralize-ai/edgerunner-android). We are
also creating specific use cases built upon Edgerunner (Llama, Stable
diffusion, etc.) which will come with their own Android bindings.

Please request additional features or desired use cases through Github issues or
on our [Discord](https://discord.gg/y9EzZEkwbR).

## 🔌 Support

### OS

|  Android |  iOS  | Linux |  MacOS  | Windows |
|:--------:|:-----:|:-----:|:-------:|:-------:|
|    ✅    |   ⏳  |   ✅  |    ⏳   |   ⏳    |

### NPU

| Apple  | Qualcomm | MediaTek | Samsung | Intel | AMD |
|:------:|:--------:|:--------:|:-------:|:-----:|:---:|
|   ⏳   |    ✅    |    ⏳    |   ⏳    |  ⏳   | ⏳  |

## 🛠 Building and installing

See the [BUILDING](BUILDING.md) document.

## 🕹 Usage

See [examples](example/README.md) for basic usage instructions.

## 🏆 Contributing

See the [CONTRIBUTING](CONTRIBUTING.md) document.

Join our [Discord](https://discord.gg/y9EzZEkwbR) for discussing any issues.

## 📜 Licensing

See the [LICENSING](LICENSE.txt) document.
