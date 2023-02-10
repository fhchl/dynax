from abc import abstractmethod
from typing import Optional, Union, Callable

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

from .interpolation import spline_it
from .system import DynamicalSystem


try:
    # TODO: remove when upgrading to diffrax > v0.2
    DefaultAdjoint = dfx.NoAdjoint
except AttributeError:
    DefaultAdjoint = dfx.DirectAdjoint


class AbstractEvolution(eqx.Module):
    """Abstract base-class for evolutions."""

    @abstractmethod
    def __call__(self, x0, t, **kwargs):
        raise NotImplementedError


class Flow(AbstractEvolution):
    """Evolution function for continous-time dynamical system."""

    system: DynamicalSystem
    solver: dfx.AbstractAdaptiveSolver = eqx.static_field(default_factory=dfx.Dopri5)
    step: dfx.AbstractStepSizeController = eqx.static_field(
        default_factory=dfx.ConstantStepSize
    )
    dt0: Optional[float] = eqx.static_field(default=None)

    def __call__(
        self,
        x0: ArrayLike,
        t: ArrayLike,
        u: Optional[Union[ArrayLike, Callable[[float], float]]] = None,
        ucoeffs: Optional[ArrayLike] = None,
        squeeze=True,
        **diffeqsolve_kwargs,
    ):
        """Solve initial value problem for state and output trajectories."""
        t = jnp.asarray(t)
        x0 = jnp.asarray(x0)
        assert (
            len(x0) == self.system.n_states
        ), f"len(x0)={len(x0)} but sys has {self.system.n_states} states"

        if u is None and ucoeffs is None:
            ufun = lambda t: None
        elif ucoeffs is not None:
            path = dfx.CubicInterpolation(t, ucoeffs)
            ufun = path.evaluate
        elif callable(u):
            ufun = u
        else:
            ufun = spline_it(t, u)

        # Solve ODE
        diffeqsolve_default_options = dict(
            solver=self.solver,
            stepsize_controller=self.step,
            saveat=dfx.SaveAt(ts=t),
            max_steps=50 * len(t),
            adjoint=DefaultAdjoint(),
            dt0=self.dt0 if self.dt0 is not None else t[1],
        )
        diffeqsolve_default_options |= diffeqsolve_kwargs
        vector_field = lambda t, x, self: self.system.vector_field(x, ufun(t), t)
        term = dfx.ODETerm(vector_field)
        x = dfx.diffeqsolve(
            term,
            t0=t[0],
            t1=t[-1],
            y0=x0,
            args=self,  # https://github.com/patrick-kidger/diffrax/issues/135
            **diffeqsolve_default_options,
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

    def __call__(
        self,
        x0: ArrayLike,
        num_steps: Optional[int] = None,
        t: Optional[ArrayLike] = None,
        u: Optional[ArrayLike] = None,
        squeeze: bool = True,
    ):
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
