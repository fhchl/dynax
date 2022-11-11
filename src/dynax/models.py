import jax.numpy as jnp

from dynax import DynamicalSystem, ControlAffine


class SpringMassDamper(DynamicalSystem):
  """Forced second-order spring-mass-damper system: m*x'' + r*x' + k*x = u. """
  m: float
  r: float
  k: float
  def vector_field(self, x, u=None, t=None):
    if u is None: u = 0
    x1, x2 = x
    return jnp.array([x2, (u - self.r*x2 - self.k*x1)/self.m])


class Sastry9_9(ControlAffine):
  """Sastry Example 9.9"""
  n_states = 3
  n_inputs = 1
  n_params = 0
  def f(self, x, t=None): return jnp.array([0., x[0] + x[1]**2, x[0] - x[1]])
  def g(self, x, t=None): return jnp.array([jnp.exp(x[1]), jnp.exp(x[1]), 0.])
  def h(self, x, t=None): return x[2]