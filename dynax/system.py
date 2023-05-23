"""Classes for representing dynamical systems."""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from .util import ssmatrix


def _linearize(f, h, x0, u0, t):
    """Linearize dx=f(x,u), y=h(x,u) around x0, u0."""
    A = jax.jacfwd(f, argnums=0)(x0, u0, t)
    B = jax.jacfwd(f, argnums=1)(x0, u0, t)
    C = jax.jacfwd(h, argnums=0)(x0, u0, t)
    D = jax.jacfwd(h, argnums=1)(x0, u0, t)
    return A, B, C, D


class DynamicalSystem(eqx.Module):
    r"""A continous-time dynamical system.

    .. math::

        ẋ &= f(x, u, t) \\
        y &= h(x, u, t)

    """

    # these attributes should be overridden by subclasses
    n_states: int = eqx.static_field(default=None, init=False)
    n_inputs: int = eqx.static_field(default=None, init=False)
    n_outputs: int = eqx.static_field(default=None, init=False)

    # Don't know if it is possible to set vector_field and output
    # in a __init__ method, which would make the API nicer. For
    # now, this class must always be subclassed.
    # As a an attribute, it can't be assigned to during init.
    # As a eqx.static_field, it is not supported by jax, as the JIT
    # compiler doesn't support staticmethods.
    def vector_field(self, x, u=None, t=None):
        """Compute state derivative."""
        raise NotImplementedError

    def output(self, x, u=None, t=None):
        """Compute output."""
        return x

    def linearize(self, x0=None, u0=None, t=None) -> "LinearSystem":
        """Compute the approximate linearized system around a point."""
        if x0 is None:
            x0 = np.zeros(self.n_states)
        if u0 is None:
            u0 = np.zeros(self.n_inputs)
        A, B, C, D = _linearize(self.vector_field, self.output, x0, u0, t)
        # jax creates empty arrays
        if B.size == 0:
            B = np.zeros((self.n_states, self.n_inputs))
        if D.size == 0:
            D = np.zeros((C.shape[0], self.n_inputs))
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


# TODO: have output_internals, that makes methods return tuple
#       (x, pytree_interal_states_x)


class LinearSystem(DynamicalSystem):
    r"""A linear, time-invariant dynamical system.

    .. math::

        ẋ &= Ax + Bu \\
        y &= Cx + Du

    """

    # TODO: could be subclass of control-affine? Two blocking problems:
    # - right now, control affine is SISO only for io-linearization
    # - may h depend on u? Needed for D. If so, then one could compute
    #   relative degree.

    A: jnp.ndarray
    B: jnp.ndarray
    C: jnp.ndarray
    D: jnp.ndarray

    def __init__(self, A: Array, B: Array, C: Array, D: Array):
        """
        Args:
            A, B, C, D: system matrices of proper dimensions

        """
        A = ssmatrix(A)
        C = ssmatrix(C)
        B = ssmatrix(B)
        D = ssmatrix(D)
        assert A.ndim == B.ndim == C.ndim == D.ndim == 2
        assert A.shape[0] == A.shape[1]
        assert B.shape[0] == A.shape[0]
        assert C.shape[1] == A.shape[0]
        assert D.shape[1] == B.shape[1]
        assert D.shape[0] == C.shape[0]
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.n_states = A.shape[0]
        self.n_inputs = B.shape[1]
        self.n_outputs = C.shape[0]

    def vector_field(self, x, u=None, t=None):
        x = jnp.atleast_1d(x)
        out = self.A.dot(x)
        if u is not None:
            u = jnp.atleast_1d(u)
            out += self.B.dot(u)
        return out

    def output(self, x, u=None, t=None):
        x = jnp.atleast_1d(x)
        out = self.C.dot(x)
        if u is not None:
            u = jnp.atleast_1d(u)
            out += self.D.dot(u)
        return out


class ControlAffine(DynamicalSystem):
    r"""A control-affine dynamical system.

    .. math::

        ẋ &= f(x) + g(x)u \\
        y &= h(x)

    """

    def f(self, x, t=None):
        raise NotImplementedError

    def g(self, x, t=None):
        raise NotImplementedError

    def h(self, x, t=None):
        return x

    def vector_field(self, x, u=None, t=None):
        if u is None:
            u = 0
        return self.f(x, t) + self.g(x, t) * u

    def output(self, x, u=None, t=None):
        return self.h(x, t)


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
        y &= h(x, u, t)

    """

    _sys: DynamicalSystem
    _sys2: DynamicalSystem
    _feedbacklaw: Callable[[Array, Array, Float], Float]

    def __init__(
        self,
        sys: DynamicalSystem,
        sys2: DynamicalSystem,
        v: Callable[[Array, Array, Array], Array],
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
        self.n_outputs = sys.n_outputs

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
