from abc import abstractmethod
from collections.abc import Callable

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float
from .ad import extended_lie_derivative, lie_derivative
from .util import _ssmatrix


def _linearize(f, h, x0, u0):
  """Linearize dx=f(x,u), y=h(x,u) around equilibrium point."""
  A = jax.jacfwd(f, argnums=0)(x0, u0)
  B = jax.jacfwd(f, argnums=1)(x0, u0)
  C = jax.jacfwd(h, argnums=0)(x0, u0)
  D = jax.jacfwd(h, argnums=1)(x0, u0)
  return A, B, C, D


class DynamicalSystem(eqx.Module):
  # these attributes should be overridden by subclasses
  n_states: int = eqx.static_field(default=None, init=False)
  n_params: int = eqx.static_field(default=None, init=False)
  n_inputs: int = eqx.static_field(default=None, init=False)
  n_outputs: int = eqx.static_field(default=None, init=False)

  # Don't know if it is possible to set vector_field and output
  # in a __init__ method, which would make the API nicer. For
  # now, this class must always be subclassed.
  # As a an attribute, it can't be assigned to during init.
  # As a eqx.static_field, it is not supported by jax, as the JIT
  # compiler doesn't support staticmethods.
  @abstractmethod
  def vector_field(self, x, u=None, t=None):
    pass

  def output(self, x, u=None, t=None):
    """Return state by default."""
    return x

  def linearize(self, x0=None, u0=None, t=None):
    """Linearize around point."""
    if x0 is None: x0 = np.zeros(self.n_states)
    if u0 is None: u0 = np.zeros(self.n_inputs)
    A, B, C, D = _linearize(self.vector_field, self.output, x0, u0)
    # jax creates empty arrays
    if B.size == 0: B = np.zeros((self. n_states, self.n_inputs))
    if D.size == 0: D = np.zeros((C.shape[0], self.n_inputs))
    return LinearSystem(A, B, C, D)

  def obs_ident_mat(self, x0, u=None, t=None):
    """Generalized observability-identifiability matrix for constant input.

    Villaverde, 2017.
    """
    params, treedef = jax.tree_util.tree_flatten(self)

    def f(x, p):
      """Vector-field for argumented state vector xp = [x, p]."""
      model = treedef.unflatten(p)
      return model.vector_field(x, u, t)

    def g(x, p):
      """Output function for argumented state vector xp = [x, p]."""
      model = treedef.unflatten(p)
      return model.output(x, t)

    params = jnp.array(params)
    O_i = jnp.vstack(
      [jnp.hstack(
        jax.jacfwd(lie_derivative(f, g, n), (0, 1))(x0, params))
        for n in range(self.n_states+self.n_params)])

    return O_i

  def extended_obs_ident_mat(self, x0, u, t=None):
    """Generalized observability-identifiability matrix for constant input.

    Villaverde, 2017.
    """
    params, treedef = jax.tree_util.tree_flatten(self)

    def f(x, u, p):
      """Vector-field for argumented state vector xp = [x, p]."""
      model = treedef.unflatten(p)
      return model.vector_field(x, u, t)

    def g(x, p):
      """Output function for argumented state vector xp = [x, p]."""
      model = treedef.unflatten(p)
      return model.output(x, t)

    params = jnp.array(params)
    u = jnp.array(u)
    lies = [extended_lie_derivative(f, g, n) for n in range(self.n_states+self.n_params)]
    grad_of_outputs = [jnp.hstack(jax.jacfwd(l, (0, 2))(x0, u, params)) for l in lies]
    O_i = jnp.vstack(grad_of_outputs)
    return O_i

  def test_observability():
    pass

# TODO: have output_internals, that makes methods return tuple (x, pytree_interal_states_x)

class SeriesSystem(DynamicalSystem):
  """Two systems in series."""
  _sys1: DynamicalSystem
  _sys2: DynamicalSystem

  def __init__(self, sys1, sys2):
    self._sys1 = sys1
    self._sys2 = sys2
    self.n_params = sys1.n_params + sys2.n_params
    self.n_states = sys1.n_states + sys2.n_states
    self.n_inputs = sys1.n_inputs

  def vector_field(self, x, u=None, t=None):
    x1 = x[:self._sys1.n_states]
    x2 = x[self._sys1.n_states:]
    y1 = self._sys1.output(x1, u, t)
    dx1 = self._sys1.vector_field(x1, u, t)
    dx2 = self._sys2.vector_field(x2, y1, t)
    return jnp.concatenate((jnp.atleast_1d(dx1), jnp.atleast_1d(dx2)))

  def output(self, x, u=None, t=None):
    x1 = x[:self._sys1.n_states]
    x2 = x[self._sys1.n_states:]
    y1 = self._sys1.output(x1, u, t)
    y2 = self._sys2.output(x2, y1, t)
    return y2


class FeedbackSystem(DynamicalSystem):
  """Two systems in parallel."""
  _sys1: DynamicalSystem
  _sys2: DynamicalSystem

  def __init__(self, sys1, sys2):
    """sys1.output must not depend on input."""
    self._sys1 = sys1
    self._sys2 = sys2
    self.n_params = sys1.n_params + sys2.n_params
    self.n_states = sys1.n_states + sys2.n_states
    self.n_inputs = sys1.n_inputs

  def vector_field(self, x, u=None, t=None):
    if u is None: u = np.zeros(self._sys1.n_inputs)
    x1 = x[:self._sys1.n_states]
    x2 = x[self._sys1.n_states:]
    y1 = self._sys1.output(x1, None, t)
    y2 = self._sys2.output(x2, y1, t)
    dx1 = self._sys1.vector_field(x1, u+y2, t)
    dx2 = self._sys2.vector_field(x2, y1, t)
    dx = jnp.concatenate((jnp.atleast_1d(dx1), jnp.atleast_1d(dx2)))
    return dx

  def output(self, x, u=None, t=None):
    x1 = x[:self._sys1.n_states]
    y = self._sys1.output(x1, None, t)
    return y


class StaticStateFeedbackSystem(DynamicalSystem):
  _sys: DynamicalSystem
  _feedbacklaw: Callable

  def __init__(self, sys, law):
    """sys1.output must not depend on input."""
    self._sys = sys
    self._feedbacklaw = staticmethod(law)
    self.n_params = sys.n_params
    self.n_states = sys.n_states
    self.n_inputs = sys.n_inputs

  def vector_field(self, x, u=None, t=None):
    if u is None: u = np.zeros(self._sys.n_inputs)
    v = self._feedbacklaw(x, u)  # NOTE: how to extract the modified input v?
    dx = self._sys.vector_field(x, v, t)
    return dx

  def output(self, x, u=None, t=None):
    y = self._sys.output(x, None, t)
    return y

class DynamicStateFeedbackSystem(DynamicalSystem):
  _sys: DynamicalSystem
  _sys2: DynamicalSystem
  _feedbacklaw: Callable[[Array, Array, Float], Float]

  def __init__(self, sys, sys2, law):
    """Feedback u(x, z, r)"""
    self._sys = sys
    self._sys2 = sys2
    self._feedbacklaw = staticmethod(law)
    self.n_params = sys.n_params
    self.n_states = sys.n_states
    self.n_inputs = sys.n_inputs
    self.n_outputs = sys.n_outputs

  def vector_field(self, x, u=None, t=None):
    if u is None: u = np.zeros(self._sys.n_inputs)
    x, z = x[:self.n_states], x[self.n_states:]
    dx = self._sys.vector_field(x, self._feedbacklaw(x, z, u), t)
    dz = self._sys2.vector_field(z, u, t)
    return jnp.concatenate((dx, dz))

  def output(self, x, u=None, t=None):
    y = self._sys.output(x, u, t)
    return y


class LinearSystem(DynamicalSystem):
  """TODO: could be control-affine? Two blocking problems:
  - right now, control affine is SISO only
  - may h depend on u? Needed for D.
  If so, then one could compute relative degree.
  """
  A: jnp.ndarray
  B: jnp.ndarray
  C: jnp.ndarray
  D: jnp.ndarray

  def __init__(self, A, B, C, D):
    A = _ssmatrix(A)
    C = _ssmatrix(C)
    B = _ssmatrix(B)
    D = _ssmatrix(D)
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
    self.n_params = A.size + B.size + C.size + C.size
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
  @abstractmethod
  def f(self, x, t=None):
    pass

  @abstractmethod
  def g(self, x, t=None):
    pass

  def h(self, x, t=None):
    return x

  def vector_field(self, x, u=None, t=None):
    if u is None: u = 0
    return self.f(x, t) + self.g(x, t)*u

  def output(self, x, u=None, t=None):
    return self.h(x, t)


def spline_it(t, u):
  """Compute interpolating cubic-spline function."""
  u = jnp.asarray(u)
  assert len(t) == u.shape[0], 'time and input must have same number of samples'
  coeffs = dfx.backward_hermite_coefficients(t, u)
  cubic = dfx.CubicInterpolation(t, coeffs)
  fun = lambda t: cubic.evaluate(t)
  return fun


class ForwardModel(eqx.Module):
  """Combines a dynamical system with a solver."""
  system: DynamicalSystem
  solver: dfx.AbstractAdaptiveSolver = eqx.static_field()
  step: dfx.AbstractStepSizeController = eqx.static_field()

  def __init__(self, system, solver=None, step=None):
    self.system = system
    self.solver = solver if solver is not None else dfx.Dopri5()
    self.step = step if step is not None else dfx.ConstantStepSize()

  def __call__(self, x0, t, u=None, squeeze=True, **diffeqsolve_kwargs):
    """Solve dynamics for state and output trajectories."""
    t = jnp.asarray(t)
    x0 = jnp.asarray(x0)
    if u is None:
      ufun = lambda t: None
    elif callable(u):
      ufun = u
    else:  # u is array_like of shape (time, inputs)
      ufun = spline_it(t, u)
    # Solve ODE
    diffeqsolve_options = dict(saveat=dfx.SaveAt(ts=t), max_steps=50*len(t),
                               adjoint=dfx.NoAdjoint())
    diffeqsolve_options |= diffeqsolve_kwargs
    vector_field = lambda t, x, _: self.system.vector_field(x, ufun(t), t)
    term = dfx.ODETerm(vector_field)
    x = dfx.diffeqsolve(term, self.solver, t0=t[0], t1=t[-1], dt0=t[1], y0=x0,
                        stepsize_controller=self.step,
                        **diffeqsolve_options).ys
    # Compute output
    y = jax.vmap(self.system.output)(x)
    # Remove singleton dimensions
    if squeeze:
      x = x.squeeze()
      y = y.squeeze()
    return x, y


class DiscreteForwardModel(eqx.Module):
  """Compute flow map for discrete dynamical system."""
  system: DynamicalSystem

  def __call__(self, x0, num_steps=None, t=None, u=None, squeeze=True):
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
    # Remove singleton dimensions
    if squeeze:
      x = x.squeeze()
      y = y.squeeze()
    return x, y