# Hacking

Here are some instructions to help you build and test this project as a
developer and potential contributor.

Edgerunner is designed to support (cross-)compiling for many architectures and
operating systems, including most dependencies. In order to take as much of the
burden as possible away from end users, there is some complex logic to the build
process. This document is a good starting point to get up and running with CPU
support. The steps must be followed carefully, but once properly configured
allows for quick development iteration.

The [examples document](/example/README.md) outlines steps to add GPU
and NPU support, but should be attempted after successfully running the tests
as described below.

## Developer mode

Build system targets that are only useful for developers of this project are
hidden if the `edgerunner_DEVELOPER_MODE` option is disabled. Enabling this
option makes tests and other developer targets and options available. Not
enabling this option means that you are a consumer of this project and thus you
have no need for these targets and options.

Developer mode is always set to on in CI workflows.

### Presets

This project makes use of [presets][1] to simplify the process of configuring
the project. As a developer, you are recommended to always have the [latest
CMake version][2] installed to make use of the latest Quality-of-Life
additions.

You have a few options to pass `edgerunner_DEVELOPER_MODE` to the configure
command, but this project prefers to use presets.

As a developer, you should create a `CMakeUserPresets.json` file at the root of
the project:

```json
{
  "version": 2,
  "cmakeMinimumRequired": {
    "major": 3,
    "minor": 14,
    "patch": 0
  },
  "configurePresets": [
    {
      "name": "dev",
      "binaryDir": "${sourceDir}/build/dev",
      "inherits": ["dev-mode", "conan", "ci-<os>"],
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Debug"
      }
    }
  ],
  "buildPresets": [
    {
      "name": "dev",
      "configurePreset": "dev",
      "configuration": "Debug"
    }
  ],
  "testPresets": [
    {
      "name": "dev",
      "configurePreset": "dev",
      "configuration": "Debug",
      "output": {
        "outputOnFailure": true
      }
    }
  ]
}
```

You should replace `<os>` in your newly created presets file with the name of
the operating system you have, which may be `win64`, `linux` or `darwin`. You
can see what these correspond to in the
[`CMakePresets.json`](CMakePresets.json) file.

`CMakeUserPresets.json` is also the perfect place in which you can put all
sorts of things that you would otherwise want to pass to the configure command
in the terminal.

> **Note**
> Some editors are pretty greedy with how they open projects with presets.
> Some just randomly pick a preset and start configuring without your consent,
> which can be confusing. Make sure that your editor configures when you
> actually want it to, for example in CLion you have to make sure only the
> `dev-dev preset` has `Enable profile` ticked in
> `File > Settings... > Build, Execution, Deployment > CMake` and in Visual
> Studio you have to set the option `Never run configure step automatically`
> in `Tools > Options > CMake` **prior to opening the project**, after which
> you can manually configure using `Project > Configure Cache`.

### Dependency manager

The above preset will make use of the [conan][conan] dependency manager. After
installing it, make sure you have a [Conan profile][profile] setup, then
download the dependencies and generate the necessary CMake files by running
this command in the project root:

```sh
conan install . -s build_type=Debug -b missing
```

Note that if your conan profile does not specify the same compiler, standard
level, build type and runtime library as CMake, then that could potentially
cause issues. See the link above for profiles documentation.

An example Android profile is bundled with this repository. Install it to
your local conan prefix using:

```sh
conan config install profiles -tf profiles
```

and invoke it with `-pr android` in your `conan install` invocation.

[conan]: https://conan.io/
[profile]: https://docs.conan.io/2/reference/config_files/profiles.html

### Configure, build and test

If you followed the above instructions, then you can configure, build and test
the project respectively with the following commands from the project root on
any operating system with any build system:

```sh
cmake --preset=dev
cmake --build --preset=dev
ctest --preset=dev
```

If you are using a compatible editor (e.g. VSCode) or IDE (e.g. CLion, VS), you
will also be able to select the above created user presets for automatic
integration.

Please note that both the build and test commands accept a `-j` flag to specify
the number of jobs to use, which should ideally be specified to the number of
threads your CPU has. You may also want to add that to your preset using the
`jobs` property, see the [presets documentation][1] for more details.

For Android, the above `ctest` approach does not work. Instead, provided that `conan install` is invoked with an appropriate android profile and Android compatible presets are used, there will be an additional `test-android` target that can be executed with:

```sh
cmake --build --preset=<preset> -t test-android
```

Ensure [adb](https://developer.android.com/tools/adb) is configured and a device
with USB debugging enabled is connected.

### Developer mode targets

These are targets you may invoke using the build command from above, with an
additional `-t <target>` flag:

#### `coverage`

Available if `ENABLE_COVERAGE` is enabled. This target processes the output of
the previously run tests when built with coverage configuration. The commands
this target runs can be found in the `COVERAGE_TRACE_COMMAND` and
`COVERAGE_HTML_COMMAND` cache variables. The trace command produces an info
file by default, which can be submitted to services with CI integration. The
HTML command uses the trace command's output to generate an HTML document to
`<binary-dir>/coverage_html` by default.

#### `docs`

Available if `BUILD_MCSS_DOCS` is enabled. Builds to documentation using
Doxygen and m.css. The output will go to `<binary-dir>/docs` by default
(customizable using `DOXYGEN_OUTPUT_DIRECTORY`).

#### `format-check` and `format-fix`

These targets run the clang-format tool on the codebase to check errors and to
fix them respectively. Customization available using the `FORMAT_PATTERNS` and
`FORMAT_COMMAND` cache variables.

#### run-examples

Available if `-o examples=True` was supplied to the `conan install` invocation.
This is because the examples may require additional dependencies for pre and post
processing that we do not wish the bundle with the main project.

Runs all the examples created by the `add_example` command.

Individual examples can be executed using `run_<example_name>` (without the
extension) instead of `run-examples`.

See [examples](./example) for more details.

#### `spell-check` and `spell-fix`

These targets run the codespell tool on the codebase to check errors and to fix
them respectively. Customization available using the `SPELL_COMMAND` cache
variable.

[1]: https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html
[2]: https://cmake.org/download/
