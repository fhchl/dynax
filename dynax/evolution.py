from typing import Callable, Optional

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax._src.config import _validate_default_device
from jaxtyping import Array, ArrayLike, PyTree

from .interpolation import spline_it
from .system import DynamicalSystem
from .util import broadcast_right, dim2shape


try:
    # TODO: remove when upgrading to diffrax > v0.2
    DefaultAdjoint = dfx.NoAdjoint
except AttributeError:
    DefaultAdjoint = dfx.DirectAdjoint


class AbstractEvolution(eqx.Module):
    """Abstract base-class for evolutions."""

    def __call__(self, x0: ArrayLike, t: ArrayLike, u: ArrayLike, **kwargs):
        raise NotImplementedError


def check_shape(shape, dim, arg):
    if (
        not (dim == "scalar" and shape != ()) and 
        not (shape[1:] == (dim,))
    ):
        raise ValueError(f"Argument {arg} of shape {shape} is size {dim}.")
    

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
        u: Optional[ArrayLike] = None,
        ufun: Optional[Callable[[float], Array]] = None,
        ucoeffs: Optional[tuple[PyTree, PyTree, PyTree, PyTree]] = None,
        squeeze: bool = True,
        **diffeqsolve_kwargs,
    ) -> tuple[Array, Array]:
        """Solve initial value problem for state and output trajectories."""
        t = jnp.asarray(t)
        x0 = jnp.asarray(x0)

        # Check initial state shape
        if x0.shape != dim2shape(self.system.n_states):
            raise ValueError("Initial state dimenions do not match.")

        # Prepare input function
        if u is None and ufun is None and ucoeffs is None and self.system.n_inputs == 0:
            _ufun = lambda t: jnp.empty((0,))
        elif ucoeffs is not None:
            path = dfx.CubicInterpolation(t, ucoeffs)
            _ufun = path.evaluate
        elif callable(u):
            _ufun = u
        elif u is not None:
            u = jnp.asarray(u)
            if len(t) != u.shape[0]:
                raise ValueError("t and u must have matching first dimension.")
            _ufun = spline_it(t, u)
        else:
            raise ValueError("Must specify one of u, ufun, or ucoeffs.")

        # Check shape of ufun return values
        out = jax.eval_shape(_ufun, 0.)
        if not isinstance(out, jax.ShapeDtypeStruct):
            raise ValueError(f"ufun must return Arrays, not {type(out)}.")
        else:
            if not out.shape == dim2shape(self.system.n_inputs):
                raise ValueError("Input dimensions do not match.")

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
        vector_field = lambda t, x, self: self.system.vector_field(x, _ufun(t), t)
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
        y = jax.vmap(self.system.output)(x, u, t)

        # Remove singleton dimensions
        if squeeze:
            x = x.squeeze()
            y = y.squeeze()

        return x, y


class Map(AbstractEvolution):
    """Flow map for evolving a discrete-time dynamical system."""

    system: DynamicalSystem

    def __call__(
        self,
        x0: ArrayLike,
        t: Optional[Array] = None,
        u: Optional[Array] = None,
        num_steps: Optional[int] = None,
        squeeze: bool = True,
    ):
        """Solve discrete map."""
        x0 = jnp.asarray(x0)

        if t is not None:
            t = jnp.asarray(t)
            num_steps = len(t)
        elif u is not None:
            u = jnp.asarray(u)
            num_steps = len(u)
        elif num_steps is not None:
            t = jnp.zeros(num_steps)
        else:
            raise ValueError("must specify one of num_steps, t, or u.")

        if t is not None and u is not None:
            if t.shape[0] != u.shape[0]:
                raise ValueError("t and u must have the same first dimension.")
            inputs = jnp.stack((broadcast_right(t, u), u), axis=1)
            unpack = lambda input: (input[0], input[1])
        elif t is not None:
            inputs = t
            unpack = lambda input: (input, None)
        else:
            inputs = u
            unpack = lambda input: (None, input)

        def scan_fun(state, input):
            t, u = unpack(input)
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
