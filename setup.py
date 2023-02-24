from distutils.core import setup
from Cython.Build   import cythonize

import numpy

if __name__ == "__main__":
    setup(ext_modules=cythonize("*.pyx", compiler_directives={"language_level": "3"}), include_dirs=[numpy.get_include()])