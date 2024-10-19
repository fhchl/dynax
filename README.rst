Dynax
=====

*"Dynamical systems in JAX"*

|workflow_badge| |doc_badge|

.. |workflow_badge| image:: https://github.com/fhchl/dynax/actions/workflows/run_tests.yml/badge.svg
   :target: https://github.com/fhchl/dynax/actions/workflows/run_tests.yml
.. |doc_badge| image:: https://readthedocs.org/projects/dynax/badge/?version=latest
   :target: https://dynax.readthedocs.io/en/latest/?badge=latest

**This is WIP. Expect things to break!**

This package allows for straight-forward simulation, fitting and linearization of dynamical systems
by combing `JAX`_, `Diffrax`_, `Equinox`_, and `scipy.optimize`_. Its main features
include:

- estimation of ODE parameters and their covariance via the prediction-error method (`example <examples/fit_nonlinear_ode.ipynb>`_)
- estimation of the initial state (`example <examples/fit_initial_state.py>`_)
- estimation of linear ODE parameters via matching of frequency-response functions (`example <examples/fit_long_input.py>`_)
- estimation from multiple experiments
- estimation with a poor man's multiple shooting (`example <examples/fit_multiple_shooting.py>`_)
- input-output linearization of continuous-time input affine systems
- input-output linearization of discrete-time systems (`example <examples/linearize_recurrent_network.py>`_)
- estimation of a system's relative-degree (`example <examples/linearize_recurrent_network.py>`_)

Documentation is on its way. Until then, have a look at the `example <examples>`_ and `test <tests>`_ folders.


Installing
----------

Requires Python 3.9+, JAX 0.4.23+, Equinox 0.11+ and Diffrax 0.5+. With a
suitable version of jaxlib installed:

::

    pip install .


Testing
-------

Install with

::

    pip install .[dev]

and run

::

    pytest

To also test the examples, do

::

    pytest --runslow


Related software
----------------

- `nlgreyfast`_: Matlab library for fitting ODE's with mutliple shooting
- `dynamax`_: inference and learning for probablistic state-space models

.. _scipy.optimize: https://docs.scipy.org/doc/scipy/reference/optimize.html
.. _dynamax: https://github.com/probml/dynamax
.. _nlgreyfast: https://github.com/meco-group/nlgreyfast
.. _jax: https://github.com/google/jax
.. _diffrax: https://github.com/patrick-kidger/diffrax
.. _equinox: https://github.com/patrick-kidger/equinox
