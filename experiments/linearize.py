import dynax as dx
import equinox as eqx
import numpy as np
import jax.numpy as jnp
import diffrax as dfx
import matplotlib.pyplot as plt

class LoudspeakerDynamics(dx.ControlAffine):
  Bl: float
  Re: float
  Rm: float
  K: float
  L: float
  M: float
  outputs: list = eqx.static_field()

  def __init__(self, params, outputs=[0,]):
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

n = 1000
sr = 96000
t = jnp.array(np.arange(n)/sr)
u = jnp.array(np.random.normal(size=n))
coeffs = dfx.backward_hermite_coefficients(t, u)
cubic = dfx.CubicInterpolation(t, coeffs)
ufun = lambda t: cubic.evaluate(t)
#initial_params = [4., 3., 2., 1000., 1e-3, 10e-3] # With these barely, controllable test fails
initial_params = [1., 2., 3., 4., 5, 6]
dyn = LoudspeakerDynamics(initial_params)
true_model = dx.ForwardModel(dyn, sr)
x0 = jnp.array([0., 0., 0.]).T
y = true_model(t, x0, ufun)

compensator, estimator_sys = dyn.feedback_linearize(x0)

#estimator should behave same as original system as both linear
estimator = dx.ForwardModel(estimator_sys, sr)

y_true = true_model(t, x0, ufun)
y_lin_est = estimator(t, x0, ufun)

for i in range(10):
  print(compensator(i, x0))

plt.figure()
plt.plot(t, y_true)
plt.plot(t, y_lin_est)
plt.show()

