#!/usr/bin/env python

from setuptools import dist
dist.Distribution().fetch_build_eggs(['cython', 'numpy'])

import os
import platform
import numpy as np

from Cython.Build import cythonize
from setuptools import find_packages
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
        result = subprocess.call([os.environ.get("CC") or 'cc', '-fopenmp', '-lgomp', filename],
                                 stdout=fnull, stderr=fnull)
    os.chdir(curdir)
    #clean up
    shutil.rmtree(tmpdir)

    return result == 0

def _check_avx2():
    import os, tempfile, subprocess, shutil    
    avx2_test = \
        r"""
#include <immintrin.h>
    int main() {
        __m256i color_swap_mask =  _mm256_set_epi32(
            7, 7, 7, 7,
            6, 4, 2, 0
        );
        __m256i m = _mm256_slli_epi32(color_swap_mask, 1);
        return 0;
    }
"""
    tmpdir = tempfile.mkdtemp()
    curdir = os.getcwd()
    os.chdir(tmpdir)
    filename = r'test_avx2.c'
    with open(filename, 'w') as file:
        file.write(avx2_test)
    with open(os.devnull, 'w') as fnull:
        result = subprocess.call([os.environ.get("CC") or 'cc', '-mavx2', filename],
                                 stdout=fnull, stderr=fnull)
    os.chdir(curdir)
    #clean up
    shutil.rmtree(tmpdir)
    return result == 0


extra_compile_args = []
extra_link_args = ["-lstdc++"]
if platform.system() != 'Windows':
    if _check_openmp():
        extra_compile_args.append('-fopenmp')
        extra_link_args.append('-lgomp')

    if _check_avx2():
        extra_compile_args.append("-DUSE_AVX2")
        # extra_compile_args.append("-DFAST_SLIC_AVX2_FASTER")
        extra_compile_args.append("-mavx2")



setup(
    name="fast-slic",
    version="0.2.0",
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
                "cfast_slic",
                include_dirs=[np.get_include()],
                sources=["fast-slic.cpp", "fast-slic-avx2.cpp", "cfast_slic.pyx"],
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
                language='c++11',
            ),
        ]
    )
)

