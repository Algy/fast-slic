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

# https://stackoverflow.com/questions/11013851/speeding-up-build-process-with-distutils
import os
import multiprocessing
try:
    from concurrent.futures import ThreadPoolExecutor as Pool
except ImportError:
    from multiprocessing.pool import ThreadPool as LegacyPool

    # To ensure the with statement works. Required for some older 2.7.x releases
    class Pool(LegacyPool):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()
            self.join()


def build_extensions(self):
    """Function to monkey-patch
    distutils.command.build_ext.build_ext.build_extensions

    """
    self.check_extensions_list(self.extensions)

    try:
        num_jobs = os.cpu_count()
    except AttributeError:
        num_jobs = multiprocessing.cpu_count()

    with Pool(num_jobs) as pool:
        results = pool.map(self.build_extension, self.extensions)
        list(results)

def compile(
    self, sources, output_dir=None, macros=None, include_dirs=None,
    debug=0, extra_preargs=None, extra_postargs=None, depends=None,
):
    """Function to monkey-patch distutils.ccompiler.CCompiler"""
    macros, objects, extra_postargs, pp_opts, build = self._setup_compile(
        output_dir, macros, include_dirs, sources, depends, extra_postargs
    )
    cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)

    for obj in objects:
        try:
            src, ext = build[obj]
        except KeyError:
            continue
        self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

    # Return *all* object filenames, not just the ones we just built.
    return objects


from distutils.ccompiler import CCompiler
from distutils.command.build_ext import build_ext
build_ext.build_extensions = build_extensions


def _compile_and_check(c_content, compiler_args = []):
    import os, tempfile, subprocess, shutil
    tmpdir = tempfile.mkdtemp()
    curdir = os.getcwd()
    os.chdir(tmpdir)
    filename = r'test.c'
    with open(filename, 'w') as file:
        file.write(c_content)
    with open(os.devnull, 'w') as fnull:
        args = [os.environ.get("CC") or 'cc']
        args += compiler_args
        args.append(filename)
        result = subprocess.call(args, stdout=fnull, stderr=fnull)
    os.chdir(curdir)
    #clean up
    shutil.rmtree(tmpdir)
    return result == 0

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
    return _compile_and_check(omp_test, ['-fopenmp', '-lgomp'])

def _check_avx2():
    from cpuid.cpuid import CPUID
    try:
        # Invoke CPUID instruction with eax 0x7
        # ECX bit 5: AVX2 support
        # For more information, refer to https://en.wikipedia.org/wiki/CPUID
        input_eax = 0x7
        output_eax, output_ebx, output_ecx, output_edx = CPUID()(input_eax)
        bits = bin(output_ebx)[::-1]
        avx2_support = bits[5]
        return avx2_support == '1'
    except:
        return False

def _check_neon():
    neon_test = r"""
#include <arm_neon.h>
    int main() {
        const uint16x8_t constant = { 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF };
        const uint32x4_t vadded = vpaddlq_u16(constant);
        return 0;
    }
    """
    return _compile_and_check(neon_test, ["-mfpu=neon"])



extra_compile_args = []
extra_link_args = []
if platform.system() != 'Windows':
    extra_compile_args.append("-std=c++11")
    if _check_openmp():
        extra_compile_args.append('-fopenmp')
        extra_link_args.append('-lgomp')

    if _check_avx2():
        extra_compile_args.append("-DUSE_AVX2")
        extra_compile_args.append("-mavx2")
        extra_compile_args.append("-mfma")
    if platform.machine().lower().startswith("arm") and _check_neon():
        extra_compile_args.append("-DUSE_NEON")
        extra_compile_args.append("-mfpu=neon")

else:
    extra_compile_args.append("/openmp")
    if _check_avx2():
        extra_compile_args.append("/DUSE_AVX2")
        extra_compile_args.append("/arch:AVX2")
        extra_compile_args.append("/fp:fast")


cpp_sources = [
    "src/timer.cpp",
    "src/parallel.cpp",
    "src/fast-slic.cpp",
    "src/cca.cpp",
    "src/context.cpp",
    "src/context-impl.cpp",
    "src/lsc.cpp",
    "src/lsc-builder.cpp",
]


setup(
    name="fast-slic",
    version="0.4.0",
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
                sources=cpp_sources + ["cfast_slic.pyx"],
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
                language="c++",
            ),
            Extension(
                "csimple_crf",
                include_dirs=[np.get_include()],
                sources=["src/simple-crf.cpp", "csimple_crf.pyx"],
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
                language="c++",
            )
        ]
    )
)
