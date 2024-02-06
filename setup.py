import re
from glob import glob
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, find_packages

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
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    author='Josh Meyers',
    author_email='jmeyers314@gmail.com',
    url='https://github.com/jmeyers314/danish',
    description="Geometric donut engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['danish'],
    package_data={
        "danish": ["data/*"],
    },
    install_requires=['pybind11>2.10', 'numpy', 'pyyaml', 'galsim', 'batoid', 'scipy'],
    python_requires='>=3.8',
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
