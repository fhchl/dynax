"""Classes for representing dynamical systems."""

from abc import abstractmethod
from collections.abc import Callable
from dataclasses import field
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from equinox.internal import ω
from jax import Array
from jax.tree_util import tree_map, tree_structure
from jax.typing import ArrayLike
from jaxtyping import PyTree, ScalarLike


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


from jax._src.numpy.util import promote_dtypes_numeric

def _linearize(f, h, x, u, t):
    """Linearize dx=f(x,u,t), y=h(x,u,t) around x, u and t."""
    A = jax.jacfwd(f, argnums=0)(x, u, t)
    B = jax.jacfwd(f, argnums=1)(x, u, t) if u is not None else None
    C = jax.jacfwd(h, argnums=0)(x, u, t)
    D = jax.jacfwd(h, argnums=1)(x, u, t) if u is not None else None
    return A, B, C, D


class DynamicalSystem(eqx.Module):
    r"""A continous-time dynamical system with optional output equation.

    .. math::

        ẋ &= f(x, u, t) \\
        y &= h(x, u, t)

    Subclasses must implement f via the `vector_field` method, which must return PyTrees
    of the same structure as `x`. One may optionally override the
    `output` method to implement h, which should return a `PyTree` or `None`.

    `DynamicalSystem` inherits from `equinox.Module` which is a dataclass, so an
    `__init__` method is not necessarily required.

    Example::

        class IntegratorAndGain(DynamicalSystem):
            gain: float

            def vector_field(self, x, u, t):
                dx = u
                return dx

            def output(self, x, u, t):
                return self.gain*x

    """

    x0: Optional[PyTree] = static_field(init=False, default=None)

    @abstractmethod
    def vector_field(
        self, x: PyTree, u: Optional[ScalarLike] = None, t: Optional[ScalarLike] = None
    ) -> PyTree:
        """Compute the state derivative from current state, input and time."""
        pass

    def output(
        self, x: PyTree, u: Optional[ScalarLike] = None, t: Optional[ScalarLike] = None
    ) -> PyTree:
        """Compute the output from current state, input and time."""
        return None

    def linearize(
        self,
        x: Optional[PyTree] = None,
        u: Optional[ScalarLike] = None,
        t: Optional[float] = None,
    ) -> "LinearSystem":
        """Compute the linearized system around a state, and input and time."""
        x = self.x0 if x is None else x
        if x is None:
            raise ValueError("Specify either x or set x0 attribute.")
        A, B, C, D = _linearize(self.vector_field, self.output, x, u, t)
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
    #   pass


def _control_affine(
    bfun: Callable[[PyTree], PyTree],
    Afun: Callable[[PyTree], PyTree],
    x: PyTree,
    u: Optional[ScalarLike] = None
):
    b = bfun(x)
    A = Afun(x)
    if u is None or A is None:
        return b
    # TODO(pytree): some casting to arrays is needed here, as otherwise the omegas behave
    # weired. For np.ndarray a, a**ω is ω applied for each element resulting in a array
    # of object
    return (b**ω + A**ω * u).ω


class ControlAffine(DynamicalSystem):
    r"""A control-affine dynamical system.

    .. math::

        ẋ &= f(x) + g(x)u \\
        y &= h(x) + i(x)u

    """

    def f(self, x: PyTree) -> PyTree:
        raise NotImplementedError

    def g(self, x: PyTree) -> PyTree:
        raise NotImplementedError

    def h(self, x: PyTree) -> Array | None:
        return None

    def i(self, x: PyTree) -> Array | None:
        return None

    def vector_field(self, x, u=None, t=None):
        return _control_affine(self.f, self.g, x, u)

    def output(self, x, u=None, t=None):
        return _control_affine(self.h, self.i, x, u)


class LinearSystem(ControlAffine):
    r"""A linear, time-invariant dynamical system.

    .. math::

        ẋ &= Ax + Bu \\
        y &= Cx + Du

    """
    A: PyTree
    B: Optional[PyTree] = None
    C: Optional[PyTree] = None
    D: Optional[PyTree] = None

    def f(self, x: PyTree) -> PyTree:
        return tree_map(jnp.dot, self.A, x)

    def g(self, x: PyTree) -> PyTree:
        return self.B

    def h(self, x: PyTree) -> PyTree:
        return tree_map(jnp.dot, self.C, x)

    def i(self, x: PyTree) -> PyTree:
        return self.D


class LinearFuncSystem(ControlAffine):
    r"""A linear, time-invariant dynamical system.

    .. math::

        ẋ &= Ax + Bu \\
        y &= Cx + Du

    """
    A: Callable
    B: Optional[Callable] = None
    C: Optional[Callable] = None
    D: Optional[Callable] = None

    def f(self, x: PyTree) -> PyTree:
        return self.A(x)

    def g(self, x: PyTree) -> PyTree:
        return self.B(x)

    def h(self, x: PyTree) -> PyTree:
        return tree_map(jnp.dot, self.C, x)

    def i(self, x: PyTree) -> PyTree:
        return self.D


class SeriesSystem(DynamicalSystem):
    """Two systems in series."""

    sys1: DynamicalSystem
    sys2: DynamicalSystem

    def vector_field(self, x, u=None, t=None):
        x1, x2 = x
        y1 = self.sys1.output(x1, u, t)
        dx1 = self.sys1.vector_field(x1, u, t)
        dx2 = self.sys2.vector_field(x2, y1, t)
        return dx1, dx2

    def output(self, x, u=None, t=None):
        x1, x2 = x
        y1 = self.sys1.output(x1, u, t)
        y2 = self.sys2.output(x2, y1, t)
        return y2


class FeedbackSystem(DynamicalSystem):
    """Two systems connected via feedback."""

    sys: DynamicalSystem
    feedbacksys: DynamicalSystem

    def vector_field(self, x, u=None, t=None):
        x1, x2 = x
        y1 = self.sys.output(x1, None, t)
        y2 = self.feedbacksys.output(x2, y1, t)
        dx1 = self.sys.vector_field(x1, u + y2, t)
        dx2 = self.feedbacksys.vector_field(x2, y1, t)
        return dx1, dx2

    def output(self, x, u=None, t=None):
        x1, _ = x
        y = self.sys.output(x1, None, t)
        return y


class StaticStateFeedbackSystem(DynamicalSystem):
    r"""System with static state-feedback.

    .. math::

        ẋ &= f(x, v(x, u), t) \\
        y &= h(x, u, t)

    """

    sys: DynamicalSystem
    feedbacklaw: Callable[[PyTree, PyTree], PyTree]

    def __init__(self, sys: DynamicalSystem, v: Callable[[PyTree, PyTree], PyTree]):
        """
        Args:
            sys: system with vector field `f` and output `h`
            v: static feedback law `v`

        """
        self.sys = sys
        self.feedbacklaw = staticmethod(v)

    def vector_field(self, x, u=None, t=None):
        v = self.feedbacklaw(x, u)
        dx = self.sys.vector_field(x, v, t)
        return dx

    def output(self, x, u=None, t=None):
        y = self.sys.output(x, u, t)
        return y


class DynamicStateFeedbackSystem(DynamicalSystem):
    r"""System with dynamic state-feedback.

    .. math::

        ẋ &= f_1(x, v(x, z, u), t) \\
        ż &= f_2(z, r, t) \\
        y &= h(x, u, t)

    """

    sys: DynamicalSystem
    sys2: DynamicalSystem
    feedbacklaw: Callable[[Array, Array, float], float]

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
        self.sys = sys
        self.sys2 = sys2
        self.feedbacklaw = v

    def vector_field(self, x, u=None, t=None):
        x, z = x
        v = self.feedbacklaw(x, z, u)
        dx = self.sys.vector_field(x, v, t)
        dz = self.sys2.vector_field(z, u, t)
        return (dx, dz)

    def output(self, x, u=None, t=None):
        x, _ = x
        y = self.sys.output(x, u, t)
        return y
