from abc import abstractmethod
from collections.abc import Callable

import equinox as eqx
import diffrax as dfx
from scipy.optimize import least_squares
import numpy as np

import jax
import jax.numpy as jnp
from functools import lru_cache
from dynax.util import MemoizeJac, value_and_jacfwd


class DynamicalSystem(eqx.Module):
  n_states: int = eqx.static_field()
  n_params: int = eqx.static_field()

  @abstractmethod
  def vector_field(self, x, u=None, t=None):
    pass

  @abstractmethod
  def output(self, x, t=None):
    pass

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
    O_i = jnp.vstack([jnp.hstack(jax.jacfwd(lie_derivative(f, g, n), (0, 1))(x0, params)) for n in range(self.n_states+self.n_params)])

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

class LinearSystem(DynamicalSystem):
  A: jnp.ndarray
  B: jnp.ndarray
  C: jnp.ndarray

  def __init__(self, A, B, C):
    assert A.shape[0] == A.shape[1]
    assert B.shape[0] == A.shape[0]
    assert A.ndim == B.ndim == C.ndim == 2
    assert C.shape[1] == A.shape[0]
    self.A = A
    self.B = B
    self.C = C
    self.n_states = A.shape[0]
    self.n_params = len(A.reshape(-1)) + len(B.reshape(-1)) + len(C.reshape(-1))

  def vector_field(self, x, u, t=None):
    assert x.ndim == 1
    x = x[:, None]
    return (self.A.dot(x) + self.B.dot(u)).squeeze()

  def output(self, x, t=None):
    return self.C.dot(x)

  def linearize(self, x0):
    return self

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

  def output(self, x, t=None):
    return self.h(x, t)

  def linearize(self, x0=None) -> LinearSystem:
    if x0 is None:
      x0 = np.zeros((self.n_states, 1))
    A = jax.jacfwd(self.f)(x0)
    B = self.g(x0)[:, None]
    C = jax.jacfwd(self.h)(x0)
    return LinearSystem(A, B, C)

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

@lru_cache
def lie_derivative(f, h, n=1):
  """Returns function for n-th derivative of h along f.

  ..math: L_f^n h(x) = (\nabla L_f^{n-1} h)(x)^T f(x)
          L_f^0 h(x) = h(x)

  """
  if n==0:
    return h
  else:
    return lambda x, *args: jax.jvp(h, (x, *args), (f(x, *args),))[1]

@lru_cache
def extended_lie_derivative(f, h, n=1):
  """Returns function for n-th derivative of h along f.

  ..math: L_f^n h(x, t) = (\nabla_x L_f^{n-1} h)(x, t)^T f(x, u, t)
          L_f^0 h(x, t) = h(x, t)

  """
  if n==0:
    return lambda x, _, p: h(x, p)
  elif n==1:
    return lambda x, u, p: jax.jacfwd(h)(x, p).dot(f(x, u[0], p))
    #return lambda x, u, p: jax.jvp(h, (x, p), (f(x, u[0], p), ))[1] # FIXME: Tree structure of primal and tangential must be the same
  else:
    last_lie = extended_lie_derivative(f, h, n-1)
    grad_x = jax.jacfwd(last_lie, 0)
    grad_u = jax.jacfwd(last_lie, 1)
    def fun(x, u, p):
      uterms = min(n-2, len(u)-1)
      return (grad_x(x, u, p).dot(f(x, u[0], p)) +
              grad_u(x, u, p)[:, :uterms].dot(u[1:uterms+1]))
    return fun

def fit_ml(model: ForwardModel, t, u, y, x0):
  """Fit forward model via maximum likelihood."""
  coeffs = dfx.backward_hermite_coefficients(t, u)
  cubic = dfx.CubicInterpolation(t, coeffs)
  ufun = lambda t: cubic.evaluate(t)
  init_params, treedef = jax.tree_flatten(model)
  std_y = np.std(y, axis=0)

  # scale parameters and bounds
  def residuals(params):
    model = treedef.unflatten(params)
    pred_y, _ = model(t, x0, ufun)
    res = ((y - pred_y)/std_y).reshape(-1)
    return res / np.sqrt(len(res))

  # compute primal and sensitivties in one forward pass
  fun = MemoizeJac(jax.jit(lambda x: value_and_jacfwd(residuals, x)))
  jac = fun.derivative
  res = least_squares(fun, init_params, jac=jac, x_scale='jac', verbose=2)
  return res.x
