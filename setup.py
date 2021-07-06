import os
import re
import sys
import subprocess

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion

VERSIONFILE="danish/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    version = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))


with open("README.rst", 'r') as fh:
    long_description = fh.read()


setup(
    name='danish',
    version=version,
    author='Josh Meyers',
    author_email='jmeyers314@gmail.com',
    url='https://github.com/jmeyers314/danish',
    description="Geometric donut engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['danish'],
    package_dir={'danish': 'danish'},
    install_requires=['numpy'],
    python_requires='>=3.6',
    zip_safe=False,
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
