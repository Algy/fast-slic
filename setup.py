#!/usr/bin/env python

import os
import numpy as np
import platform

from setuptools import find_packages
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extra_compile_args = []
if platform.system() != 'Windows':
    extra_compile_args += ['-std=c++11']


setup(
    name="fast-slic",
    version="0.1.0",
    description="Fast Slic Superpixel Implementation",
    author="Alchan Kim",
    author_email="a9413miky@gmail.com",
    setup_requires = ["cython", "numpy"],
    install_requires=["numpy"],
    python_requires=">=3.5",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: Implementation :: CPython",
    ],
    packages=find_packages(),
    ext_modules=cythonize(
        [
            Extension(
                "libfastslic",
                sources=["fast-slic.cpp"],
                extra_compile_args=extra_compile_args,
                language='c++11',
            ),
            Extension(
                "cfast_slic",
                sources=["cfast_slic.pyx"],
                include_dirs=[np.get_include()],
            )
        ],
    )
)

