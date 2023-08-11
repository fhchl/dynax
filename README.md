# Dynax

_"Dynamical systems in JAX"_

[![Run tests](https://github.com/fhchl/dynax/actions/workflows/run_tests.yml/badge.svg)](https://github.com/fhchl/dynax/actions/workflows/run_tests.yml)

__This is WIP!__

This package combines [JAX][jax], [Diffrax][diffrax], and [Equinox][equinox] for
straight-forward simulation, fitting and linearization of dynamical systems. See
[example](examples) and [test](tests) folders for some documentation.


## Installing

Requires Python 3.9+, JAX 0.4.13+, Equinox 0.10.10+ and Diffrax 0.4.0+.

With the correct version of jaxlib installed:

    pip install .


## Testing

Install with

    pip install .[dev]

and run

    pytest

To also test the examples, do

    pytest --runslow


[jax]: https://github.com/google/jax
[diffrax]: https://github.com/patrick-kidger/diffrax
[equinox]: https://github.com/patrick-kidger/equinox
