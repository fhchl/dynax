"""Classes for representing dynamical systems."""

from abc import abstractmethod
from collections.abc import Callable
from dataclasses import field
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import ArrayLike

from .util import dim2shape


def _linearize(f, h, x0, u0, t0):
    """Linearize dx=f(x,u,t), y=h(x,u,t) around x0, u0, t0."""
    A = jax.jacfwd(f, argnums=0)(x0, u0, t0)
    B = jax.jacfwd(f, argnums=1)(x0, u0, t0)
    C = jax.jacfwd(h, argnums=0)(x0, u0, t0)
    D = jax.jacfwd(h, argnums=1)(x0, u0, t0)
    return A, B, C, D


def static_field(**kwargs):
    """Like `equinox.static_field`, but removes constraints if they exist."""
    try:
        metadata = dict(kwargs["metadata"])
    except KeyError:
        metadata = kwargs["metadata"] = {}
    if "static" in metadata:
        raise ValueError("Cannot use metadata with `static` already set.")
    metadata["static"] = True
    metadata["constrained"] = False
    return field(**kwargs)


def boxed_field(lower: float, upper: float, **kwargs):
    """Mark a field value as box-constrained."""
    try:
        metadata = dict(kwargs["metadata"])
    except KeyError:
        metadata = kwargs["metadata"] = {}
    metadata["constrained"] = ("boxed", (lower, upper))
    metadata["static"] = False
    return field(**kwargs)


def free_field(**kwargs):
    """Remove the value constraint from attribute, e.g. when subclassing."""
    try:
        metadata = dict(kwargs["metadata"])
    except KeyError:
        metadata = kwargs["metadata"] = {}
    metadata["static"] = False
    metadata["constrained"] = False
    return field(**kwargs)


def non_negative_field(min_val: float = 0.0, **kwargs):
    """Mark a parameter as non-negative."""
    return boxed_field(lower=min_val, upper=np.inf, **kwargs)


# TODO: make abstract


class DynamicalSystem(eqx.Module):
    r"""A continous-time dynamical system.

    .. math::

        ẋ &= f(x, u, t) \\
        y &= h(x, u, t)

    Subclasses must set values for attributes n_states, n_inputs, and implement the
    `vector_field` method. Use the optional `output` method to describe measurent
    equations. Otherwise, the total state is returned as output.

    In most cases, it is not needed to define a custom __init__ method, as
    `DynamicalSystem` is a dataclass.

    Example::

        class IntegratorAndGain(DynamicalSystem):
            n_states = 1
            n_inputs = "scalar"
            gain: float

            def vector_field(self, x, u, t):
                dx = u
                return dx

            def output(self, x, u, t):
                return self.gain*x

    """

    initial_state: Array = static_field(init=False)
    n_inputs: int | Literal["scalar"] = static_field(init=False)

    def __check_init__(self):
        # Check that required attributes are initialized
        required_attrs = ["initial_state", "n_inputs"]
        for attr in required_attrs:
            if not hasattr(self, attr):
                raise AttributeError(f"Attribute '{attr}' not initialized.")

        # Check that vector_field and output returns Arrays or scalars and not PyTrees
        x = self.initial_state
        u = jax.ShapeDtypeStruct(dim2shape(self.n_inputs), jnp.float64)
        try:
            dx = jax.eval_shape(self.vector_field, x, u, t=1.0)
            y = jax.eval_shape(self.output, x, u, t=1.0)
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
    def vector_field(self, x, u=None, t=None) -> Array:
        """Compute state derivative."""
        raise NotImplementedError

    def output(self, x, u=None, t=None) -> Array:
        """Compute output."""
        return x

    @property
    def n_outputs(self) -> int | Literal["scalar"]:
        # Compute output size
        x = self.initial_state
        u = jax.ShapeDtypeStruct(dim2shape(self.n_inputs), jnp.float64)
        y = jax.eval_shape(self.output, x, u, t=1.0)
        return "scalar" if y.ndim == 0 else y.shape[0]

    def linearize(self, x0=None, u0=None, t=None) -> "LinearSystem":
        """Compute the approximate linearized system around a point."""
        if x0 is None:
            x0 = self.initial_state
        if u0 is None:
            u0 = jnp.zeros(dim2shape(self.n_inputs))
        A, B, C, D = _linearize(self.vector_field, self.output, x0, u0, t)
        return LinearSystem(A, B, C, D)

    # def obs_ident_mat(self, x0, u=None, t=None):
    #   """Generalized observability-identifiability matrix for constant input.

    #   Villaverde, 2017.
    #   """
    #   params, treedef = jax.tree_util.tree_flatten(self)

    #   def f(x, p):
    #     """Vector-field for argumented state vector xp = [x, p]."""
    #     model = treedef.unflatten(p)
    #     return model.vector_field(x, u, t)

    #   def g(x, p):
    #     """Output function for argumented state vector xp = [x, p]."""
    #     model = treedef.unflatten(p)
    #     return model.output(x, t)

    #   params = jnp.array(params)
    #   O_i = jnp.vstack(
    #     [jnp.hstack(
    #       jax.jacfwd(lie_derivative(f, g, n), (0, 1))(x0, params))
    #       for n in range(self.n_states+self.n_params)])

    #   return O_i

    # def extended_obs_ident_mat(self, x0, u, t=None):
    #   """Generalized observability-identifiability matrix for constant input.

    #   Villaverde, 2017.
    #   """
    #   params, treedef = jax.tree_util.tree_flatten(self)

    #   def f(x, u, p):
    #     """Vector-field for argumented state vector xp = [x, p]."""
    #     model = treedef.unflatten(p)
    #     return model.vector_field(x, u, t)

    #   def g(x, p):
    #     """Output function for argumented state vector xp = [x, p]."""
    #     model = treedef.unflatten(p)
    #     return model.output(x, t)

    #   params = jnp.array(params)
    #   u = jnp.array(u)
    #   lies = [extended_lie_derivative(f, g, n)
    #           for n in range(self.n_states+self.n_params)]
    #   grad_of_outputs = [jnp.hstack(jax.jacfwd(l, (0, 2))(x0, u, params))
    #                      for l in lies]
    #   O_i = jnp.vstack(grad_of_outputs)
    #   return O_i

    # def test_observability():
    #   pass# This allows to us to guess the shapes of the state-space mawhen given
    # as trices 1d-arrays
    # , we interpet them asumn vectors.


# TODO: have output_internals, that makes methods return tuple
#       (x, pytree_interal_states_x)


class ControlAffine(DynamicalSystem):
    r"""A control-affine dynamical system.

    .. math::

        ẋ &= f(x) + g(x)u \\
        y &= h(x) + i(x)u

    """

    @abstractmethod
    def f(self, x: Array) -> Array:
        pass

    @abstractmethod
    def g(self, x: Array) -> Array:
        pass

    def h(self, x: Array) -> Array:
        return x

    def i(self, x: Array) -> Array:
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


class LinearSystem(ControlAffine):
    r"""A linear, time-invariant dynamical system.

    .. math::

        ẋ &= Ax + Bu \\
        y &= Cx + Du

    """

    A: Array
    B: Array
    C: Array
    D: Array

    def __init__(self, A: ArrayLike, B: ArrayLike, C: ArrayLike, D: ArrayLike):
        self.A = jnp.array(A)
        self.B = jnp.array(B)
        self.C = jnp.array(C)
        self.D = jnp.array(D)

    @property
    def initial_state(self) -> Array:  # type: ignore
        return jnp.array(0) if self.A.ndim == 0 else jnp.zeros(self.A.shape[0])

    @property
    def n_inputs(self) -> int | Literal["scalar"]:  # type: ignore
        if self.initial_state.ndim == 0:
            if self.B.ndim == 0:
                return "scalar"
            elif self.B.ndim == 1:
                return self.B.size
        else:
            if self.B.ndim == 1:
                return "scalar"
            elif self.B.ndim == 2:
                return self.B.shape[1]
        raise ValueError("Dimension mismatch.")

    def f(self, x):
        return self.A.dot(x)

    def g(self, x):
        return self.B

    def h(self, x):
        return self.C.dot(x)

    def i(self, x):
        return self.D


class _CoupledSystemMixin(eqx.Module):
    _sys1: DynamicalSystem
    _sys2: DynamicalSystem

    def _pack_states(self, x1, x2) -> Array:
        return jnp.concatenate(
            (
                jnp.atleast_1d(x1),
                jnp.atleast_1d(x2),
            )
        )

    def _unpack_states(self, x):
        sys1_size = (
            1
            if jnp.ndim(self._sys1.initial_state) == 0
            else self._sys1.initial_state.size
        )
        return (
            x[:sys1_size].reshape(self._sys1.initial_state.shape),
            x[sys1_size:].reshape(self._sys2.initial_state.shape),
        )


class SeriesSystem(DynamicalSystem, _CoupledSystemMixin):
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

    """

    def __init__(self, sys1: DynamicalSystem, sys2: DynamicalSystem):
        """
        Args:
            sys1: system with n outputs
            sys2: system with n inputs

        """
        self._sys1 = sys1
        self._sys2 = sys2
        self.initial_state = self._pack_states(sys1.initial_state, sys2.initial_state)
        self.n_inputs = sys1.n_inputs

    def vector_field(self, x, u=None, t=None):
        x1, x2 = self._unpack_states(x)
        y1 = self._sys1.output(x1, u, t)
        dx1 = self._sys1.vector_field(x1, u, t)
        dx2 = self._sys2.vector_field(x2, y1, t)
        return self._pack_states(dx1, dx2)

    def output(self, x, u=None, t=None):
        x1, x2 = self._unpack_states(x)
        y1 = self._sys1.output(x1, u, t)
        y2 = self._sys2.output(x2, y1, t)
        return y2


class FeedbackSystem(DynamicalSystem, _CoupledSystemMixin):
    r"""Two systems connected via feedback.

    .. math::

        ẋ_1 &= f_1(x_1, u + y_2, t) \\
        y_1 &= h_1(x_1, t)          \\
        ẋ_2 &= f_2(x_2, y_1, t)     \\
        y_2 &= h_2(x_2, y_1, t)     \\

    .. aafig::

               +------+
        u --+->+ sys1 +--+-> y1
            ^  +------+  |
            |            |
          y2|  +------+  |
            +--+ sys2 |<-+
               +------+

    """

    def __init__(self, sys1: DynamicalSystem, sys2: DynamicalSystem):
        """
        Args:
            sys1: system in forward path with n inputs
            sys2: system in feedback path with n outputs

        """
        self._sys1 = sys1
        self._sys2 = sys2
        self.initial_state = self._pack_states(sys1.initial_state, sys2.initial_state)
        self.n_inputs = sys1.n_inputs

    def vector_field(self, x, u=None, t=None):
        if u is None:
            u = np.zeros(dim2shape(self._sys1.n_inputs))
        x1, x2 = self._unpack_states(x)
        y1 = self._sys1.output(x1, None, t)
        y2 = self._sys2.output(x2, y1, t)
        dx1 = self._sys1.vector_field(x1, u + y2, t)
        dx2 = self._sys2.vector_field(x2, y1, t)
        dx = self._pack_states(dx1, dx2)
        return dx

    def output(self, x, u=None, t=None):
        x1, _ = self._unpack_states(x)
        y = self._sys1.output(x1, None, t)
        return y


class StaticStateFeedbackSystem(DynamicalSystem):
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

    """

    _sys: DynamicalSystem
    _v: Callable[[Array], Array]

    def __init__(self, sys: DynamicalSystem, v: Callable[[Array], Array]):
        """
        Args:
            sys: system with vector field `f` and output `h`
            v: static feedback law `v`

        """
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


class DynamicStateFeedbackSystem(DynamicalSystem, _CoupledSystemMixin):
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

    """

    _v: Callable[[Array, Array, float], float]

    def __init__(
        self,
        sys1: DynamicalSystem,
        sys2: DynamicalSystem,
        v: Callable[[Array, Array, Array | float], float],
    ):
        r"""
        Args:
            sys1: system with vector field :math:`f_1` and output :math:`h`
            sys2: system with vector field :math:`f_2`
            v: dynamic feedback law :math:`v`

        """
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
