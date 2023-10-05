from collections.abc import Callable
from typing import Optional

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree

from .interpolation import spline_it
from .system import DynamicalSystem


class AbstractEvolution(eqx.Module):
    """Abstract base-class for evolutions."""

    def __call__(self, x0: ArrayLike, t: ArrayLike, u: ArrayLike, **kwargs):
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
        x0: Optional[PyTree],
        t: ArrayLike,
        u: Optional[PyTree] = None,
        ufun: Optional[Callable[[float], PyTree]] = None,
        ucoeffs: Optional[tuple[PyTree, PyTree, PyTree, PyTree]] = None,
        **diffeqsolve_kwargs,
    ) -> tuple[PyTree, PyTree]:
        """Solve initial value problem for state and output trajectories."""
        t_ = jnp.asarray(t)
        if x0 is None and self.system.x0 is None:
            raise ValueError("One of x0 or system.x0 must be not None")
        if u is None and ufun is None and ucoeffs is None:
            _ufun = lambda t: None
        elif ucoeffs is not None:
            path = dfx.CubicInterpolation(t_, ucoeffs)
            _ufun = path.evaluate
        elif callable(u):
            _ufun = u
        elif u is not None:
            u_ = jnp.asarray(u)
            msg = "t and u must have matching first dimensions"
            assert len(t_) == u_.shape[0], msg
            _ufun = spline_it(t_, u_)
        else:
            raise ValueError("Must specify one of u, ufun, ucoeffs.")

        # Solve ODE
        diffeqsolve_default_options = dict(
            solver=self.solver,
            stepsize_controller=self.step,
            saveat=dfx.SaveAt(ts=t_),
            max_steps=50 * len(t_),
            adjoint=dfx.DirectAdjoint(),
            dt0=self.dt0 if self.dt0 is not None else t_[1],
        )
        diffeqsolve_default_options |= diffeqsolve_kwargs
        vector_field = lambda t_, x, self: self.system.vector_field(x, _ufun(t_), t_)
        term = dfx.ODETerm(vector_field)
        x = dfx.diffeqsolve(
            term,
            t0=t_[0],
            t1=t_[-1],
            y0=x0,
            args=self,  # https://github.com/patrick-kidger/diffrax/issues/135
            **diffeqsolve_default_options,
        ).ys

        # Compute output
        y = jax.vmap(self.system.output)(x, u, t_)

        return x, y


class Map(AbstractEvolution):
    """Flow map for evolving discrete-time dynamical system."""

    system: DynamicalSystem

    def __call__(
        self,
        x0: ArrayLike,
        t: Optional[ArrayLike] = None,
        u: Optional[ArrayLike] = None,
        num_steps: Optional[int] = None,
    ):
        """Solve discrete map."""
        x0 = jnp.asarray(x0)
        if num_steps is None:
            if t is not None:
                num_steps = len(np.asarray(t))
            elif u is not None:
                num_steps = len(np.asarray(u))
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

        return x, y
