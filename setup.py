#!/usr/bin/env python

import os
import platform

from setuptools import find_packages
from setuptools.command.build_ext import build_ext as _build_ext
from distutils.core import setup
from distutils.extension import Extension



def _check_openmp():
    import os, tempfile, subprocess, shutil    

    # see http://openmp.org/wp/openmp-compilers/
    omp_test = \
        r"""
#include <omp.h>
#include <stdio.h>
    int main() {
#pragma omp parallel
    printf("Hello from thread %d, nthreads %d\n", omp_get_thread_num(), omp_get_num_threads());
    }
        """
    tmpdir = tempfile.mkdtemp()
    curdir = os.getcwd()
    os.chdir(tmpdir)
    filename = r'test.c'
    with open(filename, 'w') as file:
        file.write(omp_test)
    with open(os.devnull, 'w') as fnull:
        result = subprocess.call([os.environ.get("CC") or 'cc', '-fopenmp', filename],
                                 stdout=fnull, stderr=fnull)
    os.chdir(curdir)
    #clean up
    shutil.rmtree(tmpdir)

    return result == 0


extra_compile_args = []
if platform.system() != 'Windows':
    extra_compile_args.append('-std=c++11')
    if _check_openmp():
        extra_compile_args.append('-fopenmp')


class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())




setup(
    name="fast-slic",
    version="0.1.3",
    description="Fast Slic Superpixel Implementation",
    author="Alchan Kim",
    author_email="a9413miky@gmail.com",
    setup_requires = ["setuptools>=18.0", "cython", "numpy>=1.8"],
    install_requires=["numpy>=1.8"],
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
    package_data={'': ['*.pyx', '*.pxd', '*.h', '*.cpp']},
    cmdclass={'build_ext':build_ext},
    ext_modules=[
        Extension(
            "libfastslic",
            sources=["fast-slic.cpp"],
            extra_compile_args=extra_compile_args,
            language='c++11',
        ),
        Extension(
            "cfast_slic",
            sources=["cfast_slic.pyx"],
        )
    ],
)

