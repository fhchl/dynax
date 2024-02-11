# Dynax

_"Dynamical systems in JAX"_

[![Run tests](https://github.com/fhchl/dynax/actions/workflows/run_tests.yml/badge.svg)](https://github.com/fhchl/dynax/actions/workflows/run_tests.yml)
[![Documentation Status](https://readthedocs.org/projects/dynax/badge/?version=latest)](https://dynax.readthedocs.io/en/latest/?badge=latest)

__This is WIP. Expect things to break!__

This package allows for straight-forward simulation, fitting and linearization of dynamical systems 
by combing [JAX][jax], [Diffrax][diffrax], [Equinox][equinox], and [scipy.optimize][scipy]. Its main features
include:

- estimation of ODE parameters and their covariance via the prediction-error method ([example](examples/fit_ode.ipynb))
- estimation of the initial state ([example](examples/fit_initial_state.py))
- estimation of linear ODE parameters via matching of frequency-response functions
- estimation from multiple experiments
- estimation with a poor man's multiple shooting
- input-output linearization of continuous-time input affine systems
- input-output linearization of discrete-time systems ([example](examples/linearize_discrete_time))
- estimation of a system's relative-degree

Documentation is on its way. Until then, have a look at the [example](examples) and [test](tests) folders.


## Installing

Requires Python 3.9+, JAX 0.4.23+, Equinox 0.11+ and Diffrax 0.5+. With a 
suitable version of jaxlib installed:

    pip install .


## Testing

Install with

    pip install .[dev]

and run

    pytest

To also test the examples, do

    pytest --runslow


## Related software

- [nlgreyfast][nlgreyfast]: Matlab library for fitting ODE's with mutliple shooting
- [dynamax][dynamax]: inference and learning for probablistic state-space models


[scipy]: https://docs.scipy.org/doc/scipy/reference/optimize.html
[dynamax]: https://github.com/probml/dynamax
[nlgreyfast]: https://github.com/meco-group/nlgreyfast
[jax]: https://github.com/google/jax
[diffrax]: https://github.com/patrick-kidger/diffrax
[equinox]: https://github.com/patrick-kidger/equinox
