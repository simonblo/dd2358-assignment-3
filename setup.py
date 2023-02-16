from distutils.core import setup
from Cython.Build   import cythonize

import numpy

if __name__ == "__main__":
    setup(ext_modules=cythonize("task_1_1.pyx", compiler_directives={"language_level": "3"}), include_dirs=[numpy.get_include()])