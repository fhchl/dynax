"""Classes representing dynamical systems."""

from abc import abstractmethod
from collections.abc import Callable
from dataclasses import Field
from typing import Any, Literal, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .custom_types import FloatScalarLike
from .util import dim2shape, pretty


def _linearize(f, h, x0, u0, t0):
    """Linearize dx=f(x,u,t), y=h(x,u,t) around x0, u0, t0."""
    A = jax.jacfwd(f, argnums=0)(x0, u0, t0)
    B = jax.jacfwd(f, argnums=1)(x0, u0, t0)
    C = jax.jacfwd(h, argnums=0)(x0, u0, t0)
    D = jax.jacfwd(h, argnums=1)(x0, u0, t0)
    return A, B, C, D


T = TypeVar("T")


def _to_static_array(x: T) -> np.ndarray | T:
    if isinstance(x, jax.Array):
        return np.asarray(x)
    else:
        return x


def field(**kwargs: Any) -> Field:
    """Mark an attribute value as trainable and unconstrained.

    Args:
        **kwargs: Keyword arguments passed to :py:func:`dataclasses.field`.

    """
    try:
        metadata = dict(kwargs["metadata"])
    except KeyError:
        metadata = kwargs["metadata"] = {}
    metadata["constrained"] = False
    return eqx.field(converter=jnp.asarray, **kwargs)


def static_field(**kwargs: Any) -> Field:
    """Mark an attribute value as non-trainable.

    Like :py:func:`equinox.field`, but removes constraints if they exist and converts
    JAX arrays to Numpy arrays.

    Args:
        **kwargs: Keyword arguments passed to :py:func:`eqx.field`.

    """
    try:
        metadata = dict(kwargs["metadata"])
    except KeyError:
        metadata = kwargs["metadata"] = {}
    metadata["constrained"] = False
    return eqx.field(converter=_to_static_array, **kwargs)


def boxed_field(lower: float, upper: float, **kwargs: Any) -> Field:
    """Mark an attribute value as trainable and box-constrained on `[lower, upper]`.

    Args:
        lower: Lower bound.
        upper: Upper bound.
        **kwargs: Keyword arguments passed to :py:func:`dataclasses.field`.

    """
    try:
        metadata = dict(kwargs["metadata"])
    except KeyError:
        metadata = kwargs["metadata"] = {}
    metadata["constrained"] = ("boxed", (lower, upper))
    return field(**kwargs)


def non_negative_field(min_val: float = 0.0, **kwargs: Any) -> Field:
    """Mark an attribute value as trainable and non-negative.

    Args:
        min_val: Minimum value.
        **kwargs: Keyword arguments passed to :py:func:`dataclasses.field`.

    """
    return boxed_field(lower=min_val, upper=np.inf, **kwargs)


class AbstractSystem(eqx.Module):
    r"""Base class for dynamical systems.

    Any dynamical system in Dynax must inherit from this class. Subclasses can define
    continous-time

    .. math::

        ẋ &= f(x, u, t) \\
        y &= h(x, u, t)

    or discrete-time

    .. math::

        x_{k+1} &= f(x_k, u_k, t) \\
        y_k &= h(x_k, u_k, t)

    system. The distinction between the two is only made when instances of subclasses
    are passed to objects such as :py:class:`dynax.evolution.Flow`,
    :py:class:`dynax.evolution.Map`, :py:class:`dynax.linearize.input_output_linearize`,
    or :py:class:`dynax.linearize.discrete_input_output_linearize`.

    Subclasses must set values for the `n_inputs`, and `initial_state` attributes
    and implement the `vector_field` method. The `output` method describes the measurent
    equations. By default, the full state vector is returned as output.

    Example::

        class IntegratorAndGain(AbstractSystem):
            n_states = 1
            n_inputs = "scalar"
            gain: float

            def vector_field(self, x, u, t):
                dx = u
                return dx

            def output(self, x, u, t):
                return self.gain*x


    `AbstractSystem` is a dataclass and as such defines a default constructor which can
    make it necessary to implement a custom `__init__` method.

    """

    # TODO: make these abstract vars?
    initial_state: np.ndarray = static_field(init=False)
    """Initial state vector."""
    n_inputs: int | Literal["scalar"] = static_field(init=False)
    """Number of inputs."""

    def __check_init__(self):
        # Check that required attributes are initialized
        required_attrs = ["initial_state", "n_inputs"]
        for attr in required_attrs:
            if not hasattr(self, attr):
                raise AttributeError(f"Attribute '{attr}' not initialized.")

        with jax.ensure_compile_time_eval():
            # Check that vector_field and output returns Arrays or scalars - not PyTrees
            x = jax.ShapeDtypeStruct(self.initial_state.shape, jnp.float64)
            u = jax.ShapeDtypeStruct(dim2shape(self.n_inputs), jnp.float64)
            try:
                dx = eqx.filter_eval_shape(self.vector_field, x, u, t=1.0)
                y = eqx.filter_eval_shape(self.output, x, u, t=1.0)
            except Exception as e:
                raise ValueError(
                    "Can not evaluate output shapes. Check your definitions!"
                ) from e
            for val, func in zip((dx, y), ("vector_field, output")):  # noqa: B905
                if not isinstance(val, jax.ShapeDtypeStruct):
                    raise ValueError(
                        f"{func} must return arrays or scalars, not {type(val)}"
                    )

    @abstractmethod
    def vector_field(
        self, x: Array, u: Array | None = None, t: FloatScalarLike | None = None
    ) -> Array:
        """Compute state derivative.

        Args:
            x: State vector.
            u: Optional input vector.
            t: Optional time.

        Returns:
            State derivative.

        """
        raise NotImplementedError

    def output(
        self, x: Array, u: Array | None = None, t: FloatScalarLike | None = None
    ) -> Array:
        """Compute output.

        Args:
            x: State vector.
            u: Optional input vector.
            t: Optional time.

        Returns:
            System output.

        """
        return x

    @property
    def n_outputs(self) -> int | Literal["scalar"]:
        """The size of the output vector."""
        with jax.ensure_compile_time_eval():
            x = jax.ShapeDtypeStruct(self.initial_state.shape, jnp.float64)
            u = jax.ShapeDtypeStruct(dim2shape(self.n_inputs), jnp.float64)
            y = eqx.filter_eval_shape(self.output, x, u, t=1.0)
            n_out = "scalar" if y.ndim == 0 else y.shape[0]
        return n_out

    def linearize(
        self,
        x0: Array | None = None,
        u0: Array | None = None,
        t: FloatScalarLike | None = None,
    ) -> "LinearSystem":
        """Compute the Jacobian linearizationaround a point.

        Args:
            x0: State at which to linearize. Defaults to `initial_state`.
            u0: Input at which to linearize. Defaults to zero input.
            t: Time at which to linearize.

        Returns:
            Linearized system.

        """
        if x0 is None:
            x0 = self.initial_state
        if u0 is None:
            u0 = jnp.zeros(dim2shape(self.n_inputs))
        A, B, C, D = _linearize(self.vector_field, self.output, x0, u0, t)
        return LinearSystem(A, B, C, D)

    def pretty(self) -> str:
        """Return a pretty formatted string representation.

        The string includes the constrains of all trainable parameters and the values of
        all parameters.
        """
        return pretty(self)


class AbstractControlAffine(AbstractSystem):
    r"""Base class for control-affine dynamical systems.

    Both in continuous-time

    .. math::

        ẋ &= f(x) + g(x)u \\
        y &= h(x) + i(x)u

    or the discrete-time equivalent.

    Subclasses must implement the `f` and `g` methods that characterize the vector
    field. Optionally, the `h` and `i` methods can be implemented to describe the
    measurement equations. By default, the full state vector is returned as output.

    """

    @abstractmethod
    def f(self, x: Array) -> Array:
        """The constant-input part of the vector field."""
        pass

    @abstractmethod
    def g(self, x: Array) -> Array:
        """The input-proportional part of the vector field."""
        pass

    def h(self, x: Array) -> Array:
        """The constant-input part of the output equation."""
        return x

    def i(self, x: Array) -> Array:
        """The input-proportional part of the output equation."""
        return jnp.array(0.0)

    def vector_field(self, x, u=None, t=None):
        out = self.f(x)
        if u is not None:
            out += self.g(x).dot(u)
        return out

    def output(self, x, u=None, t=None):
        out = self.h(x)
        if u is not None:
            out += self.i(x).dot(u)
        return out


class LinearSystem(AbstractControlAffine):
    r"""A linear, time-invariant dynamical system.

    .. math::

        ẋ &= Ax + Bu \\
        y &= Cx + Du

    Args:
        A, B, C, D: System matrices of appropriate shape.

    """

    A: Array
    """State matrix."""
    B: Array
    """Input matrix."""
    C: Array
    """Output matrix."""
    D: Array
    """Feedthrough matrix."""

    def __post_init__(self):
        # Without this context manager, `initial_state` will leak later
        with jax.ensure_compile_time_eval():
            self.initial_state = (
                jnp.array(0) if self.A.ndim == 0 else jnp.zeros(self.A.shape[0])
            )
            if self.initial_state.ndim == 0:
                if self.B.ndim == 0:
                    self.n_inputs = "scalar"
                elif self.B.ndim == 1:
                    self.n_inputs = self.B.size
                else:
                    raise ValueError("Dimension mismatch.")
            else:
                if self.B.ndim == 1:
                    self.n_inputs = "scalar"
                elif self.B.ndim == 2:
                    self.n_inputs = self.B.shape[1]
                else:
                    raise ValueError("Dimension mismatch.")

    def f(self, x: Array) -> Array:
        return self.A.dot(x)

    def g(self, x: Array) -> Array:
        return self.B

    def h(self, x: Array) -> Array:
        return self.C.dot(x)

    def i(self, x: Array) -> Array:
        return self.D


class _CoupledSystemMixin(eqx.Module):
    _sys1: AbstractSystem
    _sys2: AbstractSystem

    def _pack_states(self, x1: Array, x2: Array) -> Array:
        return jnp.concatenate(
            (
                jnp.atleast_1d(x1),
                jnp.atleast_1d(x2),
            )
        )

    def _unpack_states(self, x: Array) -> tuple[Array, Array]:
        sys1_size = (
            1
            if jnp.ndim(self._sys1.initial_state) == 0
            else self._sys1.initial_state.size
        )
        return (
            x[:sys1_size].reshape(self._sys1.initial_state.shape),
            x[sys1_size:].reshape(self._sys2.initial_state.shape),
        )


class SeriesSystem(AbstractSystem, _CoupledSystemMixin):
    r"""Two systems in series.

    .. math::

        ẋ_1 &= f_1(x_1, u, t)   \\
        y_1 &= h_1(x_1, u, t)   \\
        ẋ_2 &= f_2(x_2, y1, t)  \\
        y_2 &= h_2(x_2, y1, t)

    .. aafig::

               +------+      +------+
        u --+->+ sys1 +--y1->+ sys2 +--> y2
               +------+      +------+

    Args:
        sys1: System with :math:`n` outputs.
        sys2: System with :math:`n` inputs.

    """

    def __init__(self, sys1: AbstractSystem, sys2: AbstractSystem):
        self._sys1 = sys1
        self._sys2 = sys2
        self.initial_state = self._pack_states(sys1.initial_state, sys2.initial_state)
        self.n_inputs = sys1.n_inputs

    def vector_field(
        self, x: Array, u: Array | None = None, t: FloatScalarLike | None = None
    ) -> Array:
        x1, x2 = self._unpack_states(x)
        y1 = self._sys1.output(x1, u, t)
        dx1 = self._sys1.vector_field(x1, u, t)
        dx2 = self._sys2.vector_field(x2, y1, t)
        return self._pack_states(dx1, dx2)

    def output(
        self, x: Array, u: Array | None = None, t: FloatScalarLike | None = None
    ) -> Array:
        x1, x2 = self._unpack_states(x)
        y1 = self._sys1.output(x1, u, t)
        y2 = self._sys2.output(x2, y1, t)
        return y2


class FeedbackSystem(AbstractSystem, _CoupledSystemMixin):
    r"""Two systems connected via feedback.

    .. math::

        ẋ_1 &= f_1(x_1, u + y_2, t) \\
        y_1 &= h_1(x_1, t)          \\
        ẋ_2 &= f_2(x_2, y_1, t)     \\
        y_2 &= h_2(x_2, y_1, t)

    .. aafig::

               +------+
        u --+->+ sys1 +--+-> y1
            ^  +------+  |
            |            |
          y2|  +------+  |
            +--+ sys2 |<-+
               +------+

    Args:
        sys1: System in forward path with :math:`n` inputs.
        sys2: System in feedback path with :math:`n` outputs.

    """

    def __init__(self, sys1: AbstractSystem, sys2: AbstractSystem):
        self._sys1 = sys1
        self._sys2 = sys2
        self.initial_state = self._pack_states(sys1.initial_state, sys2.initial_state)
        self.n_inputs = sys1.n_inputs

    def vector_field(
        self, x: Array, u: Array | None = None, t: FloatScalarLike | None = None
    ) -> Array:
        if u is None:
            u = jnp.zeros(dim2shape(self._sys1.n_inputs))
        x1, x2 = self._unpack_states(x)
        y1 = self._sys1.output(x1, None, t)
        y2 = self._sys2.output(x2, y1, t)
        dx1 = self._sys1.vector_field(x1, u + y2, t)
        dx2 = self._sys2.vector_field(x2, y1, t)
        dx = self._pack_states(dx1, dx2)
        return dx

    def output(
        self, x: Array, u: Array | None = None, t: FloatScalarLike | None = None
    ) -> Array:
        x1, _ = self._unpack_states(x)
        y = self._sys1.output(x1, None, t)
        return y


class StaticStateFeedbackSystem(AbstractSystem):
    r"""System with static state-feedback.

    .. math::

        ẋ &= f(x, v(x), t) \\
        y &= h(x, u, t)

    .. aafig::

                           +-----+
        u --+------------->+ sys +----> y
            ^              +--+--+
            |                 |
            |                 | x
            |  +--------+     |
            +--+ "v(x)" +<----+
               +--------+

    Args:
        sys: System with vector field :math:`f` and output :math:`h`.
        v: Static feedback law :math:`v`.

    """

    _sys: AbstractSystem
    _v: Callable[[Array], Array]

    def __init__(self, sys: AbstractSystem, v: Callable[[Array], Array]):
        self._sys = sys
        self._v = staticmethod(v)
        self.initial_state = sys.initial_state
        self.n_inputs = sys.n_inputs

    def vector_field(self, x, u=None, t=None):
        v = self._v(x)
        dx = self._sys.vector_field(x, v, t)
        return dx

    def output(self, x, u=None, t=None):
        y = self._sys.output(x, u, t)
        return y


class DynamicStateFeedbackSystem(AbstractSystem, _CoupledSystemMixin):
    r"""System with dynamic state-feedback.

    .. math::

        ẋ_1 &= f_1(x_1, v(x_1, x_2, u), t) \\
        ẋ_2 &= f_2(x_2, u, t)              \\
        y   &= h_1(x_1, u, t)

    .. aafig::

              +--------------+     +-----+
        u -+->+ v(x1, x2, u) +--v->+ sys +-> y
           |  +-+-------+----+     +--+--+
           |    ^       ^             |
           |    | x2    |      x1     |
           |    |       +-------------+
           |  +------+
           +->+ sys2 |
              +------+

    Args:
        sys1: System with vector field :math:`f_1` and output :math:`h`.
        sys2: System with vector field :math:`f_2`.
        v: dynamic feedback law :math:`v`.

    """

    _v: Callable[[Array, Array, float], float]

    def __init__(
        self,
        sys1: AbstractSystem,
        sys2: AbstractSystem,
        v: Callable[[Array, Array, Array | float], float],
    ):
        self._sys1 = sys1
        self._sys2 = sys2
        self._v = staticmethod(v)
        self.initial_state = self._pack_states(sys1.initial_state, sys2.initial_state)
        self.n_inputs = sys1.n_inputs

    def vector_field(self, x, u=None, t=None):
        if u is None:
            u = np.zeros(dim2shape(self._sys1.n_inputs))
        x1, x2 = self._unpack_states(x)
        v = self._v(x1, x2, u)
        dx = self._sys1.vector_field(x1, v, t)
        dz = self._sys2.vector_field(x2, u, t)
        return jnp.concatenate((dx, dz))

    def output(self, x, u=None, t=None):
        x1, _ = self._unpack_states(x)
        y = self._sys1.output(x1, u, t)
        return y
