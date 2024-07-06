# Examples

Set up a [`CMakeUserPresets.json`](/HACKING.md#presets) and ensure
[conan](/HACKING.md#dependency-manager) is configured.

You may wish to add presets to build in `Release` mode. Add something like the
following to your `CMakeUserPresets.json`:

```json
{
  ...
  "configurePresets": [
    ...
    {
      "name": "rel",
      "binaryDir": "${sourceDir}/build/rel",
      "inherits": ["conan"],
      "generator": "Unix Makefiles",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Release"
      }
    }
  ],
  "buildPresets": [
    ...
    {
      "name": "rel",
      "configurePreset": "rel",
      "configuration": "Release",
      "jobs": "<num threads>"
    }
  ]
}
```

For MacOS, replace "Unix Makefiles" with "Xcode".

> [!NOTE]
> Examples require additional dependencies to the main library. As such, it is
required to supply `-o examples=True` to the `conan install` command.

In the examples below, for GPU support add `-o gpu=True` to the `conan install` command.
> [!NOTE]
> The tensorflow-lite conan package disables GPU by default and as such these
  steps will not work currently. I have patched the recipe locally to enable GPU
  support and will make this available on Conan Center or another repository
  soon. In the mean time, my custom recipe can be be used as outlined
  [here](https://github.com/neuralize-ai/tensorflow-lite-conan).

## Unix

Run all examples using one of the following methods from the project root directory.

For `Debug`:

```bash
conan install . -b missing -s build_type=Debug -o examples=True
cmake --preset=dev
cmake --build --preset=dev -t run-examples
```

For `Release`:

```bash
conan install . -b missing -o examples=True
cmake --preset=rel
cmake --build --preset=rel -t run-examples
```

If an existing build exists, you may need to run:

```bash
cmake --preset=rel -Dedgerunner_ENABLE_GPU=ON
```

To run an individual example, execute:

```bash
cmake --build --preset=rel -t run_<example_name>
```

where `example_name` is the example filename without the extension (eg. `mobilenet_v3_small`).

## Android

Ensure [adb](https://developer.android.com/tools/adb) is configured and a device
with USB debugging enabled is connected.

An android [conan profile](https://docs.conan.io/2/reference/config_files/profiles.html)
is required to build for Android. To use the android profile provided with this
repo, run

```bash
conan config install profiles -tf profiles
```

from the project root directory.

Using the above presets, run all examples using the following steps from the
project root directory:

```bash
conan install . -b missing -pr android -o examples=True
cmake --preset=rel
cmake --build --preset=rel -t run-examples
```

To run an individual example, execute:

```bash
cmake --build --preset=rel -t run_<example_name>
```
where `example_name` is the example filename without the extension (eg. `mobilenet_v3_small`).
