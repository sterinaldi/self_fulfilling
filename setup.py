import numpy
from setuptools import setup, find_packages
from setuptools import Extension
from setuptools.command.build_ext import build_ext as _build_ext
from codecs import open
from os import path
from distutils.extension import Extension
import os
import warnings

with open("requirements.txt") as requires_file:
    requirements = requires_file.read().split("\n")
with open("README.md") as readme_file:
    long_description = readme_file.read()

# see https://stackoverflow.com/a/21621689/1862861 for why this is here
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        self.include_dirs.append(numpy.get_include())

scripts = ['sf-p=self_fulfill.pipelines.autoregressive_p:main',
           ]

pymodules = ['self_fulfill/pipelines/autoregressive_p',
             ]

setup(
    name = 'self_fulfilling',
    author = 'Walter Del Pozzo, Stefano Rinaldi',
    author_email = 'walter.delpozzo@unipi.it, stefano.rinaldi@phd.unipi.it',
    url = 'https://github.com/sterinaldi/self_fulfilling',
    python_requires = '>=3.7',
    packages = ['self_fulfilling'],
    py_modules = pymodules,
    install_requires=requirements,
    include_dirs = ['self_fulfilling', numpy.get_include()],
    setup_requires=['numpy', 'cython'],
    package_data={"": ['*.c', '*.pyx', '*.pxd']},
    entry_points = {
        'console_scripts': scripts,
        },
    version='0.0.0',
    long_description=long_description,
    long_description_content_type='text/markdown',
    )
