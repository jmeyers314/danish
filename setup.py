import re
from glob import glob
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

VERSIONFILE="danish/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    __version__ = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

with open("README.rst", 'r') as fh:
    long_description = fh.read()

ext_modules = [
    Pybind11Extension(
        "danish._danish",
        sorted(glob("src/*.cpp")),  # Sort source files for reproducibility
    ),
]

setup(
    name='danish',
    version=__version__,
    author='Josh Meyers',
    author_email='meyers18@llnl.gov',
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
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules,
)
