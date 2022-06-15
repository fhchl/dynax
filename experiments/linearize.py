import dynax as dx
import equinox as eqx
import numpy as np
import jax.numpy as jnp
import diffrax as dfx
import matplotlib.pyplot as plt
import jax

jax.config.update("jax_enable_x64", True)


class LinearLoudspeakerDynamics(dx.ControlAffine):
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

class LoudspeakerDynamics(dx.ControlAffine):
  Bl1: float
  Bl0: float
  Re: float
  Rm: float
  K: float
  L: float
  M: float
  outputs: list = eqx.static_field()

  def __init__(self, params, outputs=[0,]):
    self.n_states = 3
    self.n_params = 7
    self.Bl1, self.Bl0, self.Re, self.Rm, self.K, self.L, self.M = params
    self.outputs = outputs

  def f(self, x, t=None):
    i, d, v = x
    Bl = (self.Bl1*d + self.Bl0)
    di = (-self.Re*i - Bl*v) / self.L
    dd = v
    dv = (Bl*i - self.Rm*v - self.K*d) / self.M
    return jnp.array([di, dd, dv])

  def g(self, x, t=None):
    di = 1 / self.L
    dd = 0
    dv = 0
    return jnp.array([di, dd, dv])

  def h(self, x, t=None):
    return x[np.array(self.outputs)]

n = 10000
sr = 96000
t = jnp.array(np.arange(n)/sr)
u = 100*jnp.array(np.random.normal(size=n))
coeffs = dfx.backward_hermite_coefficients(t, u)
cubic = dfx.CubicInterpolation(t, coeffs)
ufun = lambda t: cubic.evaluate(t)
#initial_params = [4., 3., 2., 1000., 1e-3, 10e-3] # With these barely, controllable test fails
initial_params = [100, 1., 2., 3., 1000., 1e-2, 1e-2]
dyn = LoudspeakerDynamics(initial_params, outputs=[1])
true_model = dx.ForwardModel(dyn, sr)
x0 = jnp.array([0., 0., 0.]).T

compensator, estimator_sys = dyn.feedback_linearize(x0)

estimator = dx.ForwardModel(estimator_sys, sr)

x_est = estimator(t, x0, ufun)[1]
def vector_field(t, x, _):
  u = ufun(t)
  xest = x_est.evaluate(t)
  v = compensator(u, x, xest)
  return dyn.vector_field(x, v, t)
term = dfx.ODETerm(vector_field)
saveat = dfx.SaveAt(ts=t)
sol = dfx.diffeqsolve(term, dfx.Dopri5(), t0=t[0], t1=t[-1], dt0=1/sr,
                          y0=x0, saveat=saveat, max_steps=100*len(t),
                          #stepsize_controller=dfx.ConstantStepSize()
                          )

y_lin = jax.vmap(dyn.output)(sol.ys)


for i in range(10):
  print(compensator(i, x0, x0))


y_true = true_model(t, x0, ufun)[0]
y_lin_est = estimator(t, x0, ufun)[0]

plt.figure()
plt.plot(t, y_true, label="original system")
plt.plot(t, y_lin_est, label="linearized system")
plt.plot(t, y_lin, label="feedback linearized system")
plt.legend()
plt.show()

# Linearized system and feedbacklinearized system not the same
# Next: test with much simpler system, e.g. forced nonlinear harmonic oscillator
