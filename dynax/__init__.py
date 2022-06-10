from abc import abstractmethod
from collections.abc import Callable

import equinox as eqx
import diffrax as dfx
from scipy.optimize import least_squares
import numpy as np

import jax
import jax.numpy as jnp
from functools import lru_cache


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

  def vector_field(self, x, u, t=None):
    return self.A@x + self.B*u

  def output(self, x, t=None):
    return self.C@x

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

  def linearize(self, x0=0) -> LinearSystem:
    A = jax.jacfwd(self.f)(x0)
    B = self.g(x0)
    C = jax.jacfwd(self.h)(x0)
    return LinearSystem(A, B, C)

  def feedback_linearize(self, x0: jnp.ndarray 
      ) -> tuple[Callable[[float, jnp.ndarray], float], LinearSystem]:
    # check controllalability around x0, Sastry 9.44
    linsys = self.linearize(x0)
    A, b, n = linsys.A, linsys.B, linsys.A.shape[0]
    contrmat = np.hstack([jnp.linalg.matrix_power(A, ni).dot(b) for ni in range(n)])
    if ((rank := jnp.linalg.matrix_rank(contrmat)) != n):
      raise ValueError(f"Linearized system not controllable: order={n} but rank(O)={rank}.")
      # FIXME: this raises error, even though LS system should be controllable.
    # TODO: check in volutivity of distribution, Sastry 9.42
    # Sastry 9.102
    c = linsys.C
    cAn = c.dot(jnp.linalg.matrix_power(A, n))
    cAnm1b = c.dot(jnp.linalg.matrix_power(A, n-1)).dot(b)
    Lfnh = lie_derivative(self.f, self.h, n)
    LgLfn1h = lie_derivative(self.g, lie_derivative(self.f, self.h, n-1))
    def compensator(r, x):
      return (-Lfnh(x) + cAn.dot(x) + cAnm1b*r) / LgLfn1h(x)

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

  def __call__(self, ts, x0, ufun):
    vector_field = lambda t, x, _: self.system.vector_field(x, ufun(t), t)
    term = dfx.ODETerm(vector_field)
    saveat = dfx.SaveAt(ts=ts)
    x = dfx.diffeqsolve(term, self.solver, t0=ts[0], t1=ts[-1], dt0=1/self.sr,
                             y0=x0, saveat=saveat, max_steps=100*len(ts),
                             stepsize_controller=self.step).ys
    return jax.vmap(self.system.output)(x)

@lru_cache
def lie_derivative(f, h, n=1):
  """Returns function for n-th derivative of h along f.

  ..math: L_f^n h(x) = (\nabla L_f^{n-1} h)(x)^T f(x)
          L_f^0 h(x) = h(x)

  """
  if n==0:
    return h
  else:
    grad_h = jax.jacfwd(lie_derivative(f, h, n-1))
    return lambda x, *args: grad_h(x, *args).dot(f(x, *args))

def extended_lie_derivative(f, h, n=1):
  """Returns function for n-th derivative of h along f.

  ..math: L_f^n h(x, t) = (\nabla_x L_f^{n-1} h)(x, t)^T f(x, u, t)
          L_f^0 h(x, t) = h(x, t)

  """
  if n==0:
    return lambda x, _, p: h(x, p)
  elif n==1:
    return lambda x, u, p: jax.jacfwd(h)(x, p).dot(f(x, u[0], p))
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
    pred_y = model(t, x0, ufun)
    res = ((y - pred_y)/std_y).reshape(-1)
    return res / np.sqrt(len(res))

  # solve least_squares in scaled parameter space
  fun = jax.jit(residuals)
  jac = jax.jit(jax.jacfwd(residuals))
  res = least_squares(fun, init_params, jac=jac, x_scale='jac', verbose=2)
  print(res.x)
