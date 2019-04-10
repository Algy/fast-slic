import numpy as np

from setuptools import find_packages
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

setup(
    name="fast_slic",
    version="0.1",
    description="Fast Slic Superpixel Implementation",
    author="Alchan Kim",
    author_email="a9413miky@gmail.com",
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
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    packages=find_packages(),
    ext_modules=cythonize(
        [Extension("cfast_slic", ["cfast_slic.pyx"])],
        include_dirs=[np.get_include()]
    )
)

