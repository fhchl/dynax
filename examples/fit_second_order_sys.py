"""Example: fitting a second order nonlinear system to data."""

import jax.numpy as jnp
from dynax import ControlAffine
from dynax import fit_least_squares

# Define the dynamical system
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


  # data
  t = np.linspace(0, 1, 100)
  u = np.sin(1*2*np.pi*t)
  x0 = [1., 0.]
  true_model = ForwardModel(SpringMassDamper(1., 2., 3.))
  x_true, _ = true_model(x0, t, u)
  # fit
  init_model = ForwardModel(SpringMassDamper(1., 1., 1.))
  pred_model = fit_least_squares(init_model, t, x_true, x0, u)
  # check result
  x_pred, _ = pred_model(x0, t, u)
  npt.assert_allclose(x_pred, x_true, **tols)
  npt.assert_allclose(jax.tree_util.tree_flatten(pred_model)[0],
                      jax.tree_util.tree_flatten(true_model)[0], **tols)