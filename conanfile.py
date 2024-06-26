from conan import ConanFile
from conan.tools.cmake import CMakeToolchain


class Recipe(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps", "VirtualRunEnv"

    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "gpu": [True, False],
        "examples": [True, False],
    }

    default_options = {
        "shared": False,
        "fPIC": True,
        "gpu": False,
        "examples": False,
    }

    def layout(self):
        self.folders.generators = "conan"

    def requirements(self):
        self.requires("fmt/10.2.1")
        self.requires("span-lite/0.11.0", transitive_headers=True)
        self.requires("tensorflow-lite/2.12.0")

        if self.options.examples:
            self.requires("opencv/4.9.0")

    def build_requirements(self):
        self.test_requires("catch2/3.6.0")

    def configure(self):

        if self.settings.os == "Android":
            # TODO: fix tflite Android gpu
            self.options.gpu = False

        if self.options.gpu:
            self.options["tensorflow-lite"].with_gpu = True

        if self.options.examples:
            self.options["opencv"].with_quirc = False
            self.options["opencv"].with_ffmpeg = False
            self.options["opencv"].with_tesseract = False
            self.options["opencv"].with_openexr = False
            self.options["opencv"].with_tiff = False
            self.options["opencv"].with_webp = False
            self.options["opencv"].with_msmf = False
            self.options["opencv"].with_msmf_dxva = False
            self.options["opencv"].with_opencl = True

    def generate(self):
        toolchain = CMakeToolchain(self)

        toolchain.variables["BUILD_EXAMPLES"] = self.options.examples
        toolchain.variables["edgerunner_ENABLE_GPU"] = self.options.gpu

        toolchain.generate()
