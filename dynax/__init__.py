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
    """Generalized observability-identifiability matrix.

    Villaverde, 2017.
    """
    params, treedef = jax.tree_util.tree_flatten(self)
      
    def f(xp):
      """Vector-field for argumented state vector xp = [x, p]."""
      x, p = xp[:self.n_states], xp[self.n_states:]
      model = treedef.unflatten(p)
      return jnp.concatenate((model.vector_field(x, u, t), jnp.zeros(self.n_params)))

    def g(xp):
      """Output function for argumented state vector xp = [x, p]."""
      x, p = xp[:self.n_states], xp[self.n_states:]
      model = treedef.unflatten(p)
      return model.output(x, t)
    
    xp = jnp.concatenate((x0, jnp.array(params)))
    O_i = jnp.vstack(
        [jax.jacfwd(lie_derivative(f, g, n))(xp) 
         for n in range(self.n_states+self.n_params)]
    )

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
    return lambda x: grad_h(x).dot(f(x))


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
