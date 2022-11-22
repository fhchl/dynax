import jax.numpy as jnp

from .system import ControlAffine, DynamicalSystem


class SpringMassDamper(DynamicalSystem):
  """Forced second-order spring-mass-damper system.

  .. math:: m x'' + r x' + k x = u.

  """
  m: float
  r: float
  k: float
  n_states = 2
  n_inputs = 1
  n_params = 3
  def vector_field(self, x, u=None, t=None):
    if u is None: u = 0
    x1, x2 = x
    return jnp.array([x2, (u - self.r*x2 - self.k*x1)/self.m])


class NonlinearDrag(ControlAffine):
  """Spring-mass-damper system with nonlin drag.

  .. math:: m x'' + r(x'|x'| + x') + k x = u.

  """
  m: float
  r: float
  k: float
  n_states = 2
  n_inputs = 1
  n_params = 3
  def f(self, x, u=None, t=None):
    x1, x2 = x
    return jnp.array(
      [x2, (- self.r * (x2 * jnp.abs(x2) + x2) - self.k * x1)/self.m])
  def g(self, x, u=None, t=None):
    if u is None: u = 0
    return jnp.array([0., 1./self.m])
  def h(self, x, u=None, t=None):
    return x[0]


class Sastry9_9(ControlAffine):
  """Sastry Example 9.9"""
  n_states = 3
  n_inputs = 1
  n_params = 0
  def f(self, x, t=None): return jnp.array([0., x[0] + x[1]**2, x[0] - x[1]])
  def g(self, x, t=None): return jnp.array([jnp.exp(x[1]), jnp.exp(x[1]), 0.])
  def h(self, x, t=None): return x[2]