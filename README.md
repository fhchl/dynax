# Dynax

_"Dynamical systems in JAX"_

[![Run tests](https://github.com/fhchl/dynax/actions/workflows/run_tests.yml/badge.svg)](https://github.com/fhchl/dynax/actions/workflows/run_tests.yml)

__This is WIP!__

This package combines [JAX][jax], [Diffrax][diffrax], and [Equinox][equinox] for
straight-forward simulation, fitting and linearization of dynamical systems. Main features
include:

- estimation of ODE parameters and their covariance via nonlinear Least-Squares
- estimation of linear ODE parameters via matching of transfer-functions
- fitting with multiple shooting
- input-output linearization of continuous-time input-affine systems with well-defined relative degree
- input-output linearization of discrete-time systems with well-defined relative degree
- estimation of a system's relative-degree

See [example](examples) and [test](tests) folders for some documentation. 


## Installing

Requires Python 3.9+, JAX 0.4.13+, Equinox 0.10.10+ and Diffrax 0.4.0+. With a 
suitable version of jaxlib installed:

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
