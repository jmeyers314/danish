import os
from glob import glob
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


def _version_kwarg():
    """Supply a static ``version`` only when setuptools_scm is unavailable.

    setuptools_scm is the source of truth for the version: it derives it from
    git tags (or from PKG-INFO when building an sdist) and writes
    danish/_version.py. When it's installed we pass nothing here and let its
    setuptools plugin set the metadata version -- passing ``version`` ourselves
    would suppress that (setuptools_scm skips a version that is "already set").

    But if setuptools_scm isn't importable at build time it never runs, and
    setuptools would otherwise stamp the metadata version as "0.0.0". This is
    exactly the conda-forge build: it disables pip's build isolation and omits
    setuptools_scm from the recipe's host requirements, so the conda package's
    recorded version (importlib.metadata / pip show) comes out "0.0.0" even
    though danish/_version.py is correct. The sdist always ships
    danish/_version.py, so in that case read the version back from it.
    """
    try:
        import setuptools_scm  # noqa: F401
    except ImportError:
        return {"version": _version_from_file()}
    return {}


def _version_from_file():
    import ast

    version_file = os.path.join(os.path.dirname(__file__), "danish", "_version.py")
    try:
        with open(version_file) as f:
            tree = ast.parse(f.read())
    except OSError:
        return "0.0.0"
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    return "0.0.0"


ext_modules = [
    Pybind11Extension(
        "danish._danish",
        sorted(glob("src/*.cpp")),
    ),
]

setup(
    **_version_kwarg(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
