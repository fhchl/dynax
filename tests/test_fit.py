import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from dynax import DynamicalSystem, ForwardModel, fit_ml
from dynax.models import SpringMassDamper


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