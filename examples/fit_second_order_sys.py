import jax.numpy as jnp
from dynax import ControlAffine
from dynax import fit_least_squares

class NonlinearDrag(ControlAffine):
  """Spring-mass-damper system with nonlin drag.

  .. math:: m x'' +  r x' + r2 x'|x'| + k x = u.

  """
  m: float
  r: float
  r2: float
  k: float
  n_states = 2
  n_inputs = 1
  n_outputs = 1
  def f(self, x, u=None, t=None):
    x1, x2 = x
    return jnp.array(
      [x2, (- self.r*x2 - self.r2*jnp.abs(x2)*x2 - self.k * x1)/self.m])
  def g(self, x, u=None, t=None):
    return jnp.array([0., 1./self.m])
  def h(self, x, u=None, t=None):
    return x[0]