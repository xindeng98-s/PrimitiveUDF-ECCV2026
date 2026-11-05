import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

setup(
    name="primitiveudf-custom-mc",
    ext_modules=cythonize([
        Extension(
            name="_marching_cubes_lewiner_cy",
            sources=["_marching_cubes_lewiner_cy.pyx"],
            language="c++",
            include_dirs=[np.get_include()],
            extra_compile_args=["-O3", "-Wall"],
        )
    ], compiler_directives={"language_level": "3"})
)
