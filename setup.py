import pathlib

from setuptools import find_packages, setup


_here = pathlib.Path(__file__).resolve().parent
with open(_here / "README.md", "r") as f:
    readme = f.read()


author = ("Franz M. Heuchel",)
author_email = ("franz.heuchel@pm.me",)

setup(
    name="dynax",
    version="0.0.1dev1",
    url="https://github.com/fhchl/dynax",
    description="Dynamical systems in JAX.",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    author=author,
    author_email=author_email,
    maintainer=author,
    maintainer_email=author_email,
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.20",
        "scipy>=1.9",
        "jax>=0.3,<0.4",
        "jaxlib>=0.3,<0.4",
        "diffrax>=0.2,<0.3",
        "equinox>=0.9,<0.10",
        "jaxtyping>=0.2,<0.2.13",
    ],
    extras_require={
        "dev": [
            "pytest",
            "coverage",
            "matplotlib",
            "sphinx",
            "sphinxcontrib-bibtex",
            "sphinx_rtd_theme",
            "pre-commit",
        ]
    },
)
