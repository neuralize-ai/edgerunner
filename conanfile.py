from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMake, cmake_layout
from conan.tools.files import load, copy
import os


class EdgerunnerRecipe(ConanFile):
    name = "edgerunner"
    package_type = "library"
    license = "MIT"
    author = "Ciarán O' Rourke ciaran@runlocal.ai"
    description = "Universal AI inference library for mobile devices"
    homepage = "https://runlocal.ai"
    topics = ["cpp17", "machine-learning", "neural-networks"]

    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps", "VirtualRunEnv"

    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "with_gpu": [True, False],
        "with_npu": [True, False],
        "examples": [True, False],
    }

    default_options = {
        "shared": False,
        "fPIC": True,
        "with_gpu": False,
        "with_npu": False,
        "examples": False,
    }

    exports_sources = (
        "version.txt",
        "CMakeLists.txt",
        "cmake/*",
        "include/*",
        "source/*",
    )

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def set_version(self):
        self.version = load(self, "version.txt")[:-1]

    def layout(self):
        cmake_layout(self)
        self.folders.generators = "conan"

    def requirements(self):
        self.requires("fmt/10.2.1")
        self.requires("span-lite/0.11.0", transitive_headers=True)
        self.requires("tensorflow-lite/2.12.0")

        if self.options.examples:
            self.requires("opencv/4.9.0")

        if self.options.with_npu:
            self.requires("qnn/2.23.0.24.06.24")

    def build_requirements(self):
        self.test_requires("catch2/3.6.0")

    def configure(self):
        self.options["tensorflow-lite"].with_gpu = self.options.with_gpu

        if self.options.examples:
            self.options["opencv"].with_quirc = False
            self.options["opencv"].with_ffmpeg = False
            self.options["opencv"].with_tesseract = False
            self.options["opencv"].with_openexr = False
            self.options["opencv"].with_tiff = False
            self.options["opencv"].with_webp = False
            self.options["opencv"].with_msmf = False
            self.options["opencv"].with_msmf_dxva = False
            self.options["opencv"].with_eigen = False
            self.options["opencv"].with_flatbuffers = False
            self.options["opencv"].with_protobuf = False
            self.options["opencv"].with_wayland = False
            self.options["opencv"].with_opencl = True

    def generate(self):
        toolchain = CMakeToolchain(self)

        toolchain.variables["BUILD_EXAMPLES"] = self.options.examples
        toolchain.variables["edgerunner_ENABLE_GPU"] = self.options.with_gpu
        toolchain.variables["edgerunner_ENABLE_NPU"] = self.options.with_npu

        toolchain.generate()

        if self.options.with_npu:
            qnn = self.dependencies["qnn"]
            copy(
                self,
                "*.so",
                qnn.cpp_info.components["tflite"].libdirs[0],
                os.path.join(self.source_folder, "build", "runtimeLibs"),
            )
            copy(
                self,
                "*.so",
                qnn.cpp_info.components["htp"].libdirs[0],
                os.path.join(self.source_folder, "build", "runtimeLibs"),
            )

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "edgerunner")
        self.cpp_info.set_property("cmake_target_name", "edgerunner::edgerunner")

        self.cpp_info.names["cmake_find_package"] = "edgerunner"
        self.cpp_info.names["cmake_find_package_multi"] = "edgerunner"

        defines = []

        if self.options.with_gpu:
            defines.append("edgerunner_ENABLE_GPU")

        self.cpp_info.defines = defines
        self.cpp_info.libs = ["edgerunner"]
