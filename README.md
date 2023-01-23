# Dynax

_"Dynamical systems in JAX"_

__This is WIP!__

This package combines [JAX](https://github.com/google/jax), [Diffrax](https://github.com/patrick-kidger/diffrax), and [Equinox](https://github.com/patrick-kidger/equinox) for straight-forward simulation, fitting and linearization of dynamical systems. See [example](examples) and [test](tests) folders for some documentation.

## Installing

Requires a recent version of JAX. On Windows, use WSL or the unofficial builds [here](https://github.com/cloudhan/jax-windows-builder) to install jaxlib, e.g. like this:

    pip install --find-links https://whls.blob.core.windows.net/unstable/index.html jaxlib==0.3.24 jax==0.3.24


## Testing

Install with

    pip install -e .[dev]

and run

    pytest

To also test the examples, do

    pytest --runslow