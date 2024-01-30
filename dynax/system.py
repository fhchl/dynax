"""Classes for representing dynamical systems."""

from collections.abc import Callable
from dataclasses import field
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

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
    """Remove the value constrained from attribute, e.g. when subclassing."""
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
    # these attributes should be set by subclasses
    n_states: int | Literal["scalar"] = static_field(init=False)
    n_inputs: int | Literal["scalar"] = static_field(init=False)

    def __check_init__(self):
        # Check that required attributes are initialized
        required_attrs = ["n_states", "n_inputs"]
        for attr in required_attrs:
            if not hasattr(self, attr):
                raise AttributeError(f"Attribute '{attr}' not initialized.")

        # Check that vector_field returns Arrays or scalars and not PyTrees
        x = jax.ShapeDtypeStruct(dim2shape(self.n_states), jnp.float64)
        u = jax.ShapeDtypeStruct(dim2shape(self.n_inputs), jnp.float64)
        t = 1.0
        out = jax.eval_shape(self.vector_field, x, u, t)
        if not isinstance(out, jax.ShapeDtypeStruct):
            raise ValueError(
                f"vector_field must return arrays or scalars, not {type(out)}"
            )

    @property
    def n_outputs(self) -> int | Literal["scalar"]:
        # Compute output size
        x = jax.ShapeDtypeStruct(dim2shape(self.n_states), jnp.float64)
        u = jax.ShapeDtypeStruct(dim2shape(self.n_inputs), jnp.float64)
        y = jax.eval_shape(self.output, x, u, t=1.0)
        return "scalar" if y.ndim == 0 else y.shape[0]

    def vector_field(self, x, u=None, t=None):
        """Compute state derivative."""
        raise NotImplementedError

    def output(self, x, u=None, t=None):
        """Compute output."""
        return x

    def linearize(self, x0=None, u0=None, t=None) -> "LinearSystem":
        """Compute the approximate linearized system around a point."""
        if x0 is None:
            x0 = jnp.zeros(dim2shape(self.n_states))
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


class LinearSystem(DynamicalSystem):
    r"""A linear, time-invariant dynamical system.

    .. math::

        ẋ &= Ax + Bu \\
        y &= Cx + Du

    """

    # TODO: could be subclass of control-affine? Two blocking problems:
    # - may h depend on u? Needed for D. If so, then one could compute
    #   relative degree.

    A: Array
    B: Array
    C: Array
    D: Array

    def __init__(self, A: ArrayLike, B: ArrayLike, C: ArrayLike, D: ArrayLike):
        self.A = jnp.array(A)
        self.B = jnp.array(B)
        self.C = jnp.array(C)
        self.D = jnp.array(D)

        # Extract number of states and inputs from matrices
        self.n_states = "scalar" if self.A.ndim == 0 else self.A.shape[0]
        if self.n_states == "scalar":
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

    def vector_field(self, x, u=None, t=None):
        out = self.A.dot(x)
        if u is not None:
            out += self.B.dot(u)
        return out

    def output(self, x, u=None, t=None):
        out = self.C.dot(x)
        if u is not None:
            out += self.D.dot(u)
        return out


class ControlAffine(DynamicalSystem):
    r"""A control-affine dynamical system.

    .. math::

        ẋ &= f(x) + g(x)u \\
        y &= h(x)

    """

    def f(self, x):
        raise NotImplementedError

    def g(self, x):
        raise NotImplementedError

    def h(self, x):
        return x

    # FIXME: remove time dependence
    def vector_field(self, x, u=None, t=None):
        if u is None:
            u = 0
        return self.f(x) + self.g(x) * u

    def output(self, x, u=None, t=None):
        return self.h(x)


class SeriesSystem(DynamicalSystem):
    """Two systems in series."""

    _sys1: DynamicalSystem
    _sys2: DynamicalSystem

    def __init__(self, sys1: DynamicalSystem, sys2: DynamicalSystem):
        """
        Args:
            sys1: system with n outputs
            sys2: system with n inputs
        """
        assert sys1.n_outputs == sys2.n_inputs, "in- and outputs don't match"
        self._sys1 = sys1
        self._sys2 = sys2
        self.n_states = sys1.n_states + sys2.n_states
        self.n_inputs = sys1.n_inputs

    def vector_field(self, x, u=None, t=None):
        x1 = x[: self._sys1.n_states]
        x2 = x[self._sys1.n_states :]
        y1 = self._sys1.output(x1, u, t)
        dx1 = self._sys1.vector_field(x1, u, t)
        dx2 = self._sys2.vector_field(x2, y1, t)
        return jnp.concatenate((jnp.atleast_1d(dx1), jnp.atleast_1d(dx2)))

    def output(self, x, u=None, t=None):
        x1 = x[: self._sys1.n_states]
        x2 = x[self._sys1.n_states :]
        y1 = self._sys1.output(x1, u, t)
        y2 = self._sys2.output(x2, y1, t)
        return y2


class FeedbackSystem(DynamicalSystem):
    """Two systems connected via feedback."""

    _sys: DynamicalSystem
    _fbsys: DynamicalSystem

    def __init__(self, sys: DynamicalSystem, fbsys: DynamicalSystem):
        """
        Args:
            sys: system in forward path
            fbsys: system in feedback path

        """
        self._sys = sys
        self._fbsys = fbsys
        self.n_states = sys.n_states + fbsys.n_states
        self.n_inputs = sys.n_inputs

    def vector_field(self, x, u=None, t=None):
        if u is None:
            u = np.zeros(self._sys.n_inputs)
        x1 = x[: self._sys.n_states]
        x2 = x[self._sys.n_states :]
        y1 = self._sys.output(x1, None, t)
        y2 = self._fbsys.output(x2, y1, t)
        dx1 = self._sys.vector_field(x1, u + y2, t)
        dx2 = self._fbsys.vector_field(x2, y1, t)
        dx = jnp.concatenate((jnp.atleast_1d(dx1), jnp.atleast_1d(dx2)))
        return dx

    def output(self, x, u=None, t=None):
        x1 = x[: self._sys.n_states]
        y = self._sys.output(x1, None, t)
        return y


class StaticStateFeedbackSystem(DynamicalSystem):
    r"""System with static state-feedback.

    .. math::

        ẋ &= f(x, v(x, u), t) \\
        y &= h(x, u, t)

    """

    _sys: DynamicalSystem
    _feedbacklaw: Callable

    def __init__(self, sys: DynamicalSystem, v: Callable[[Array, Array], Array]):
        """
        Args:
            sys: system with vector field `f` and output `h`
            v: static feedback law `v`

        """
        self._sys = sys
        self._feedbacklaw = staticmethod(v)
        self.n_states = sys.n_states
        self.n_inputs = sys.n_inputs

    def vector_field(self, x, u=None, t=None):
        if u is None:
            u = np.zeros(self._sys.n_inputs)
        v = self._feedbacklaw(x, u)
        dx = self._sys.vector_field(x, v, t)
        return dx

    def output(self, x, u=None, t=None):
        y = self._sys.output(x, u, t)
        return y


class DynamicStateFeedbackSystem(DynamicalSystem):
    r"""System with dynamic state-feedback.

    .. math::

        ẋ &= f_1(x, v(x, z, u), t) \\
        ż &= f_2(z, r, t) \\
        y &= h_1(x, u, t)

    """

    _sys: DynamicalSystem
    _sys2: DynamicalSystem
    _feedbacklaw: Callable[[Array, Array, float], float]

    def __init__(
        self,
        sys: DynamicalSystem,
        sys2: DynamicalSystem,
        v: Callable[[Array, Array, float], float],
    ):
        r"""
        Args:
            sys: system with vector field :math:`f_1` and output :math:`h`
            sys2: system with vector field :math:`f_2`
            v: dynamic feedback law :math:`v`

        """
        self._sys = sys
        self._sys2 = sys2
        self._feedbacklaw = v
        self.n_states = sys.n_states + sys2.n_states
        self.n_inputs = sys.n_inputs

    def vector_field(self, xz, u=None, t=None):
        if u is None:
            u = np.zeros(self._sys.n_inputs)
        x, z = xz[: self._sys.n_states], xz[self._sys.n_states :]
        v = self._feedbacklaw(x, z, u)
        dx = self._sys.vector_field(x, v, t)
        dz = self._sys2.vector_field(z, u, t)
        return jnp.concatenate((dx, dz))

    def output(self, xz, u=None, t=None):
        x = xz[: self._sys.n_states]
        y = self._sys.output(x, u, t)
        return y
