from abc import abstractmethod
from typing import Callable, cast, Optional

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .interpolation import spline_it
from .system import DynamicalSystem
from .util import broadcast_right, dim2shape


def check_shape(shape, dim, arg):
    if not (dim == "scalar" and shape != ()) and not (shape[1:] == (dim,)):
        raise ValueError(f"Argument {arg} of shape {shape} is size {dim}.")


class AbstractEvolution(eqx.Module):
    """Abstract base-class for evolutions."""

    system: DynamicalSystem

    @abstractmethod
    def __call__(
        self, t: Array, u: Optional[Array], initial_state: Optional[Array]
    ) -> tuple[Array, Array]:
        """Evolve an initial state along the vector field and compute output.

        Args:
            t: The time periode over which to solve.
            u: An optional input sequence of same length.
            initial_state: An optional, fixed initial state used instead of
                `system.initial_state`.

        """
        raise NotImplementedError


class Flow(AbstractEvolution):
    """Evolution for continous-time dynamical systems."""

    solver: dfx.AbstractAdaptiveSolver = eqx.static_field(default_factory=dfx.Dopri5)
    stepsize_controller: dfx.AbstractStepSizeController = eqx.static_field(
        default_factory=lambda: dfx.ConstantStepSize()
    )

    def __call__(
        self,
        t: Array,
        u: Optional[Array] = None,
        initial_state: Optional[Array] = None,
        *,
        ufun: Optional[Callable[[float], Array]] = None,
        ucoeffs: Optional[tuple[PyTree, PyTree, PyTree, PyTree]] = None,
        **diffeqsolve_kwargs,
    ) -> tuple[Array, Array]:
        """Solve initial value problem for state and output trajectories."""
        # Parse inputs.
        t = jnp.asarray(t)

        if initial_state is not None:
            x = jnp.asarray(initial_state)
            if initial_state.shape != self.system.initial_state.shape:
                raise ValueError("Initial state dimenions do not match.")
        else:
            initial_state = self.system.initial_state

        # Prepare input function.
        if ucoeffs is not None:
            path = dfx.CubicInterpolation(t, ucoeffs)
            _ufun = path.evaluate
        elif callable(ufun):
            _ufun = u
        elif u is not None:
            u = jnp.asarray(u)
            if len(t) != u.shape[0]:
                raise ValueError("t and u must have matching first dimension.")
            _ufun = spline_it(t, u)
        elif self.system.n_inputs == 0:
            _ufun = lambda t: jnp.empty((0,))
        else:
            raise ValueError("Must specify one of u, ufun, or ucoeffs.")

        # Check shape of ufun return values.
        out = jax.eval_shape(_ufun, 0.0)
        if not isinstance(out, jax.ShapeDtypeStruct):
            raise ValueError(f"ufun must return Arrays, not {type(out)}.")
        else:
            if not out.shape == dim2shape(self.system.n_inputs):
                raise ValueError("Input dimensions do not match.")

        # Solve ODE.
        diffeqsolve_default_options = dict(
            solver=self.solver,
            stepsize_controller=self.stepsize_controller,
            saveat=dfx.SaveAt(ts=t),
            max_steps=50 * len(t),  # completely arbitrary number of steps
            adjoint=dfx.DirectAdjoint(),
            dt0=(
                t[1]
                if isinstance(self.stepsize_controller, dfx.ConstantStepSize)
                else None
            ),
        )
        diffeqsolve_default_options |= diffeqsolve_kwargs
        vector_field = lambda t, x, self: self.system.vector_field(x, _ufun(t), t)
        term = dfx.ODETerm(vector_field)
        x = dfx.diffeqsolve(
            term,
            t0=t[0],
            t1=t[-1],
            y0=initial_state,
            args=self,  # https://github.com/patrick-kidger/diffrax/issues/135
            **diffeqsolve_default_options,
        ).ys
        # Could be in general a Pytree, but we only allow Array states.
        x = cast(Array, x)

        # Compute output.
        y = jax.vmap(self.system.output)(x, u, t)

        return x, y


class Map(AbstractEvolution):
    """Evolution for discrete-time dynamical systems."""

    def __call__(
        self,
        t: Optional[Array] = None,
        u: Optional[Array] = None,
        initial_state: Optional[Array] = None,
        *,
        num_steps: Optional[int] = None,
    ) -> tuple[Array, Array]:
        """Solve discrete map."""

        # Parse inputs.
        if initial_state is not None:
            x = jnp.asarray(initial_state)
            if initial_state.shape != self.system.initial_state.shape:
                raise ValueError("Initial state dimenions do not match.")
        else:
            initial_state = self.system.initial_state

        if t is not None:
            t = jnp.asarray(t)
        elif u is not None:
            u = jnp.asarray(u)
        elif num_steps is not None:
            t = jnp.arange(num_steps)
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

        # Evolve.
        def scan_fun(state, input):
            t, u = unpack(input)
            next_state = self.system.vector_field(state, u, t)
            return next_state, state

        _, x = jax.lax.scan(scan_fun, initial_state, inputs, length=num_steps)

        # Compute output.
        y = jax.vmap(self.system.output)(x, u, t)

        return x, y
