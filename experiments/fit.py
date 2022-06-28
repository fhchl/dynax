import jax
import dynax as dx
import equinox as eqx
import diffrax as dfx
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", False)

class LoudspeakerDynamics(dx.ControlAffine):
  Bl: float
  Re: float
  Rm: float
  K: float
  L: float
  M: float
  outputs: list = eqx.static_field()

  def __init__(self, params, outputs=[0, 2]):
    self.n_states = 3
    self.n_params = 6
    self.Bl, self.Re, self.Rm, self.K, self.L, self.M = params
    self.outputs = outputs

  def f(self, x, t=None):
    i, d, v = x
    di = (-self.Re*i - self.Bl*v) / self.L
    dd = v
    dv = (self.Bl*i - self.Rm*v - self.K*d) / self.M
    return jnp.array([di, dd, dv])

  def g(self, x, t=None):
    di = 1 / self.L
    dd = 0
    dv = 0
    return jnp.array([di, dd, dv])

  def h(self, x, t=None):
    return x[np.array(self.outputs)]

# Training data
n = 960000
sr = 96000
t = jnp.array(np.arange(n)/sr)
u = jnp.array(np.random.normal(size=n))
coeffs = dfx.backward_hermite_coefficients(t, u)
cubic = dfx.CubicInterpolation(t, coeffs)
ufun = lambda t: cubic.evaluate(t)
initial_params = [1., 1., 1., 1000., 1e-3, 1e-3]
dyn = LoudspeakerDynamics([i*2 for i in initial_params])
true_model = dx.ForwardModel(dyn, sr)
x0 = jnp.array([0., 0., 0.])
y, sol = true_model(t, x0, ufun)

# model
model = dx.ForwardModel(LoudspeakerDynamics(initial_params), sr)
init_params, treedef = jax.tree_flatten(model)
std_y = np.std(y, axis=0)

pred_params = dx.fit_ml(model, t, u, y, x0)
print(pred_params)