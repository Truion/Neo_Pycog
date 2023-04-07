#! /usr/bin/env python
"""
Setup script.

"""
import sys
from   setuptools                 import setup, find_packages, Extension
from   setuptools.command.install import install

import numpy as np

# Version warning
if sys.version_info >= (3,):
    print("Please note that this software was only tested with Python 3.9.")

# # Determine whether Cython is available
# try:
#     from Cython.Distutils import build_ext
# except ImportError:
#     print("Cython is not available.")
#     use_cython = False
# else:
#     use_cython = True

# # Build information
# if use_cython:
#     ext_modules = [Extension('pycog.euler', ['pycog/euler.pyx'],
#                              extra_compile_args=['-Wno-unused-function'],
#                              include_dirs=[np.get_include()])]
#     cmdclass    = {'build_ext': build_ext}
# else:
#     ext_modules = []
#     cmdclass    = {}

# Setup
setup(
    name='Neo_Pycog',
    version='0.2',
    license='MIT',
    author='[H. Francis Song, Guangyu R. Yang] (original pycog owners), [Mohit Mathuria] (Neo-pycog branch owner)',
    author_email='mohitmathuria786@gmail.com',
    url='https://github.com/Truion/Neo-pycog',
#     cmdclass=cmdclass,
#     ext_modules=ext_modules,
    packages=find_packages(exclude=['examples', 'examples.*', 'paper']),
    setup_requires=['numpy'],
    install_requires=['torch'],
    classifiers=[
        'Programming Language :: Python',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering'
        ]
    )
