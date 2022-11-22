import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from dynax import ForwardModel, fit_ml
from dynax.models import SpringMassDamper, LotkaVolterra

tols = dict(rtol=1e-05, atol=1e-08)

def test_fit_ml():
  # data
  t = np.linspace(0, 1, 100)
  u = np.sin(1*2*np.pi*t)
  x0 = [1., 0.]
  true_model = ForwardModel(SpringMassDamper(1., 2., 3.))
  x_true, _ = true_model(x0, t, u)
  # fit
  init_model = ForwardModel(SpringMassDamper(1., 1., 1.))
  pred_model = fit_ml(init_model, t, x_true, x0, u)
  # check result
  x_pred, _ = pred_model(x0, t, u)
  npt.assert_allclose(x_pred, x_true, **tols)
  npt.assert_allclose(jax.tree_util.tree_flatten(pred_model)[0],
                      jax.tree_util.tree_flatten(true_model)[0], **tols)


def test_fit_with_bouded_parameters():
  # data
  t = np.linspace(0, 1, 100)
  x0 = [0.5, 0.5]
  true_model = ForwardModel(LotkaVolterra(alpha=2/3, beta=4/3, gamma=1., delta=1.))
  x_true, _ = true_model(x0, t)
  # fit
  init_model = ForwardModel(LotkaVolterra(alpha=1., beta=1., gamma=1., delta=1.))
  pred_model = fit_ml(init_model, t, x_true, x0)
  # check result
  x_pred, _ = pred_model(x0, t)
  npt.assert_allclose(x_pred, x_true, **tols)
  npt.assert_allclose(jax.tree_util.tree_flatten(pred_model)[0],
                      jax.tree_util.tree_flatten(true_model)[0], **tols)
