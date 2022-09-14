from abc import abstractmethod
from collections.abc import Callable

import equinox as eqx
import diffrax as dfx
from scipy.optimize import least_squares
import numpy as np

import jax
import jax.numpy as jnp

from dynax.util import MemoizeJac, value_and_jacfwd, _ssmatrix
from dynax.ad import lie_derivative, extended_lie_derivative


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
  
  # Don't know if it is possible to set vector_field and output
  # in a __init__ method, which would make the API nicer. For
  # now, this class must always be subclassed.
  # As a an attribute, it can't be assigned to during init. 
  # As a eqx.static_field, it is not supported by jax, as the JIT
  # compiler doesn't support staticmethods.
  @abstractmethod
  def vector_field(self, x, u=None, t=None):
    pass

  @abstractmethod
  def output(self, x, u=None, t=None):
    pass

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
  """Two systems in parallel."""
  _sys: DynamicalSystem
  _feedbacklaw: Callable

  def __init__(self, sys, law):
    """sys1.output must not depend on input."""
    self._sys = sys
    self._feedbacklaw = staticmethod(law)
    self.n_params = sys.n_params
    self.n_states = sys.n_states
    self.n_inputs = sys.n_inputs
  
  def vector_field(self, x, v=None, t=None):
    if v is None: v = np.zeros(self._sys1.n_inputs)
    dx = self._sys1.vector_field(x, self._feedbacklaw(x, v), t)
    return dx

  def output(self, x, u=None, t=None):
    y = self._sys1.output(x, None, t)
    return y


class LinearSystem(DynamicalSystem):
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
  @abstractmethod
  def f(self, x, t=None):
    pass

  @abstractmethod
  def g(self, x, t=None):
    pass

  @abstractmethod
  def h(self, x, t=None):
    pass

  def vector_field(self, x, u, t=None):
    return self.f(x, t) + self.g(x, t)*u

  def output(self, x, u=None, t=None):
    return self.h(x, t)

  def feedback_linearize(self, x0: jnp.ndarray
      ) -> tuple[Callable[[float, jnp.ndarray], float], LinearSystem]:
    # check controllalability around x0, Sastry 9.44
    linsys = self.linearize(x0)
    A, b, n = linsys.A, linsys.B, linsys.A.shape[0]
    contrmat = np.hstack([jnp.linalg.matrix_power(A, ni).dot(b) for ni in range(n)])
    if ((rank := jnp.linalg.matrix_rank(contrmat)) != n):
      import warnings
      #raise ValueError(f"Linearized system not controllable: order={n} but rank(O)={rank}.")
      warnings.warn(f"Linearized system not controllable: order={n} but rank(O)={rank}.")
      # FIXME: this raises error, even though LS system should be controllable.
    # TODO: check in volutivity of distribution, Sastry 9.42
    # Sastry 9.102
    c = linsys.C
    cAn = c.dot(jnp.linalg.matrix_power(A, n))
    cAnm1b = c.dot(jnp.linalg.matrix_power(A, n-1)).dot(b)
    Lfnh = lie_derivative(self.f, self.h, n)
    LgLfn1h = lie_derivative(self.g, lie_derivative(self.f, self.h, n-1))
    def compensator(r, x, z):
      return ((-Lfnh(x) + cAn.dot(z) + cAnm1b*r) / LgLfn1h(x)).squeeze()

    return compensator, linsys


class ForwardModel(eqx.Module):
  system: eqx.Module
  sr: int = eqx.static_field()
  solver: dfx.AbstractAdaptiveSolver = eqx.static_field()
  step: dfx.AbstractStepSizeController = eqx.static_field()

  def __init__(self, system, sr, solver=dfx.Dopri5(),
               step=dfx.ConstantStepSize()):
    self.system = system
    self.sr = sr
    self.solver = solver
    self.step = step

  def __call__(self, ts, x0, ufun, dense=False):
    vector_field = lambda t, x, _: self.system.vector_field(x, ufun(t), t)
    term = dfx.ODETerm(vector_field)
    saveat = dfx.SaveAt(ts=ts, dense=dense)
    sol = dfx.diffeqsolve(term, self.solver, t0=ts[0], t1=ts[-1], dt0=1/self.sr,
                             y0=x0, saveat=saveat, max_steps=100*len(ts),
                             stepsize_controller=self.step)
    y = jax.vmap(self.system.output)(sol.ys)
    return y, sol


def fit_ml(model: ForwardModel, t, u, y, x0):
  """Fit forward model via maximum likelihood."""
  t = jnp.asarray(t)
  u = jnp.asarray(u)
  y = jnp.asarray(y)
  coeffs = dfx.backward_hermite_coefficients(t, u)
  cubic = dfx.CubicInterpolation(t, coeffs)
  ufun = lambda t: cubic.evaluate(t)
  init_params, treedef = jax.tree_util.tree_flatten(model)
  std_y = np.std(y, axis=0)

  def residuals(params):
    model = treedef.unflatten(params)
    pred_y, _ = model(t, x0, ufun)
    res = ((y - pred_y)/std_y).reshape(-1)
    return res / np.sqrt(len(res))

  # compute primal and sensitivties in one forward pass
  fun = MemoizeJac(jax.jit(lambda x: value_and_jacfwd(residuals, x)))
  jac = fun.derivative
  # use https://lmfit.github.io/lmfit-py/index.html instead?
  res = least_squares(fun, init_params, jac=jac, x_scale='jac', verbose=2)
  params = res.x
  return treedef.unflatten(params)
