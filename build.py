from torch.utils import cpp_extension
from setuptools import Extension, setup

extension = cpp_extension.CppExtension("gtl.cpp", sources=["src/cpp/gtl.cpp"])

# ext_modules = [extension]
ext_modules = []


def build(setup_kwargs):
    setup_kwargs.update(
        {
            "ext_modules": ext_modules,
            "cmdclass": {"build_ext": cpp_extension.BuildExtension},
        }
    )
