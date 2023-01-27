from abc import abstractmethod

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .interpolation import spline_it
from .system import DynamicalSystem


class AbstractEvolution(eqx.Module):
    """Abstract base-class for evolutions."""

    @abstractmethod
    def __call__(self, x0, t, **kwargs):
        raise NotImplementedError


class Flow(AbstractEvolution):
    """Evolution function for continous-time dynamical system."""

    system: DynamicalSystem
    solver: dfx.AbstractAdaptiveSolver = eqx.static_field()
    step: dfx.AbstractStepSizeController = eqx.static_field()

    def __init__(self, system, solver=dfx.Dopri5(), step=dfx.ConstantStepSize()):
        self.system = system
        self.solver = solver
        self.step = step

    def __call__(self, x0, t, u=None, squeeze=True, **diffeqsolve_kwargs):
        """Solve initial value problem for state and output trajectories."""
        t = jnp.asarray(t)
        x0 = jnp.asarray(x0)
        assert (
            len(x0) == self.system.n_states
        ), f"len(x0)={len(x0)} but sys has {self.system.n_states} states"
        if u is None:
            ufun = lambda t: None
        elif callable(u):
            ufun = u
        else:  # u is array_like of shape (time, inputs)
            ufun = spline_it(t, u)
        # Solve ODE
        diffeqsolve_options = dict(
            saveat=dfx.SaveAt(ts=t), max_steps=50 * len(t), adjoint=dfx.NoAdjoint()
        )
        diffeqsolve_options |= diffeqsolve_kwargs
        vector_field = lambda t, x, self: self.system.vector_field(x, ufun(t), t)
        term = dfx.ODETerm(vector_field)
        x = dfx.diffeqsolve(
            term,
            self.solver,
            t0=t[0],
            t1=t[-1],
            dt0=t[1],
            y0=x0,
            stepsize_controller=self.step,
            args=self,  # https://github.com/patrick-kidger/diffrax/issues/135
            **diffeqsolve_options,
        ).ys
        # Compute output
        y = jax.vmap(self.system.output)(x)
        # Remove singleton dimensions
        if squeeze:
            x = x.squeeze()
            y = y.squeeze()
        return x, y


class Map(AbstractEvolution):
    """Flow map for evolving discrete-time dynamical system."""

    system: DynamicalSystem

    def __call__(self, x0, num_steps=None, t=None, u=None, squeeze=True):
        """Solve discrete map."""
        x0 = jnp.asarray(x0)
        if num_steps is None:
            if t is not None:
                num_steps = len(t)
            elif u is not None:
                num_steps = len(u)
            else:
                raise ValueError("must specify one of num_steps, t or u")

        if t is None:
            t = np.zeros(num_steps)
        if u is None:
            u = np.zeros(num_steps)
        inputs = np.stack((t, u), axis=1)

        def scan_fun(state, input):
            t, u = input
            next_state = self.system.vector_field(state, u, t)
            return next_state, state

        _, x = jax.lax.scan(scan_fun, x0, inputs, length=num_steps)

        # Compute output
        y = jax.vmap(self.system.output)(x)
        if squeeze:
            # Remove singleton dimensions
            x = x.squeeze()
            y = y.squeeze()
        return x, y
