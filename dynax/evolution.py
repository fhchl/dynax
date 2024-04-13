from abc import abstractmethod
from typing import Callable, cast, Optional

import jax
import jax.numpy as jnp
from diffrax import (
    AbstractAdaptiveSolver,
    AbstractStepSizeController,
    ConstantStepSize,
    CubicInterpolation,
    diffeqsolve,
    DirectAdjoint,
    Dopri5,
    ODETerm,
    SaveAt,
)
from equinox import Module, static_field
from jax import Array
from jaxtyping import PyTree

from .interpolation import spline_it
from .system import AbstractSystem
from .util import broadcast_right, dim2shape


class AbstractEvolution(Module):
    """Abstract base-class for evolutions.

    from :py:class:`equinox.Module`.
    """

    system: AbstractSystem

    @abstractmethod
    def __call__(
        self, t: Array, u: Optional[Array], initial_state: Optional[Array]
    ) -> tuple[Array, Array]:
        """Evolve an initial state along the vector field and compute output.

        Args:
            t: Times at which to evaluate the evolution.
            u: An optional input sequence of same length.
            initial_state: An optional, fixed initial state used instead of
                `system.initial_state`.

        Returns:
            A tuple `(x, y)` of state and output sequences.

        """
        raise NotImplementedError


class Flow(AbstractEvolution):
    """Evolution for continous-time dynamical systems.

    Args:
        system: A dynamical system.
        solver: A diffrax solver.
        stepsize_controller: A diffrax stepsize controller.

    """

    solver: AbstractAdaptiveSolver = static_field(default_factory=Dopri5)
    stepsize_controller: AbstractStepSizeController = static_field(
        default_factory=lambda: ConstantStepSize()
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
        (
            super().__call__.__doc__
            + """

        Additional args:
            ufun: A function `t -> u` that returns the input at time `t`.
            ucoeffs: A tuple of coefficients for a cubic spline interpolation.
            **diffeqsolve_kwargs: Additional arguments to pass to `diffeqsolve`.
        """
        )
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
            path = CubicInterpolation(t, ucoeffs)
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
            saveat=SaveAt(ts=t),
            max_steps=50 * len(t),  # completely arbitrary number of steps
            adjoint=DirectAdjoint(),
            dt0=(
                t[1] if isinstance(self.stepsize_controller, ConstantStepSize) else None
            ),
        )
        diffeqsolve_default_options |= diffeqsolve_kwargs
        vector_field = lambda t, x, self: self.system.vector_field(x, _ufun(t), t)
        term = ODETerm(vector_field)
        x = diffeqsolve(
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
