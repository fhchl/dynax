import pathlib

from setuptools import setup, find_packages

_here = pathlib.Path(__file__).resolve().parent
with open(_here / "README.md", "r") as f:
    readme = f.read()


author = "Franz M. Heuchel",
author_email = "franz.heuchel@pm.me",

setup(
    name="dynax",
    version="0.0.1dev1",
    url="https://github.com/fhchl/dynax",
    description="Dynamical styems in JAX.",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    author=author,
    author_email=author_email,
    maintainer=author,
    maintainer_email=author_email,
    python_requires='>=3.7',
    install_requires=[
        'numpy',
        'jax',
        'jaxlib',
        'jaxtyping',
        'diffrax',
        'equinox',
        ],
    extras_require={
        'dev': [
            'pytest',
            'tabulate',
            'absl-py',
            'coverage'
        ]
    }
)
