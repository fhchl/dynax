import dynax as dx
import equinox as eqx
import numpy as np
import jax.numpy as jnp
import diffrax as dfx
import matplotlib.pyplot as plt
import jax
from datetime import datetime

jax.config.update("jax_enable_x64", True)

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


class LoudspeakerDynamicsFull(dx.ControlAffine):
  Bl: float | list
  Re: float
  Rm: float
  K: float | list
  L: float | list
  M: float
  L20: float
  R20: float
  Li: list
  outputs: list = eqx.static_field()

  def __init__(self, params, outputs=[0,]):
    self.n_states = None
    self.n_params = None
    self.Bl, self.Re, self.Rm, self.K, self.L, self.M, self.L20, self.R20, self.Li = params
    self.outputs = outputs

  def f(self, x, t=None):
    i, d, v, i2 = jnp.moveaxis(x, -1, 0)
    Bl = jnp.polyval(jnp.atleast_1d(self.Bl), d)
    Re = self.Re
    Rm = self.Rm
    K = jnp.polyval(jnp.atleast_1d(self.K), d)
    M = self.M
    # position and current dependent inductance
    Ld_coefs = jnp.atleast_1d(self.L)
    Li_coefs = jnp.append(jnp.atleast_1d(self.Li), 1.) # L(i) = poly + 1
    Lfun = lambda d, i: jnp.polyval(Ld_coefs, d) * jnp.polyval(Li_coefs, i) # L = L(d)L(i)
    L = Lfun(d, i)
    L_d = jax.grad(Lfun, argnums=0)(d, i) # dL/dd
    L_i = jax.grad(Lfun, argnums=1)(d, i) # dL/di
    # lossy inductance with co-variant elemts
    L20 = self.L20
    R20 = self.R20
    L0 = Ld_coefs[-1] # == L(d=0, i=0)
    L2 = L20 * L/L0
    R2 = R20 * L/L0
    L2_d = L20 * L_d/L0
    L2_i = L20 * L_i/L0
    # state evolution
    di = ((Re + R2)*i + R2*i2 - (Bl + L_d*i)*v) / (L + i*L_i)
    dd = v
    dv = ((Bl + 0.5*(L_d*i + L2_d*i2))*i - Rm*v - K*d) / M
    di2 = (R2 * (i - i2) - L2_d*i2*v) / (L2 + i*L2_i)
    return jnp.array([di, dd, dv, di2])

  def g(self, x, t=None):
    i, d, _, _ = x
    Ld_coefs = jnp.atleast_1d(self.L)
    Li_coefs = jnp.append(jnp.atleast_1d(self.Li), 1.) # L(i) = poly + 1
    Lfun = lambda d, i: jnp.polyval(Ld_coefs, d) * jnp.polyval(Li_coefs, i) # L = L(d)L(i)
    L = Lfun(d, i)
    L_i = jax.grad(Lfun, argnums=1)(d, i) # dL/di
    return jnp.array([1/(L + i*L_i), 0., 0., 0.])

  def h(self, x, t=None):
    return x[np.array(self.outputs)]


def modelSimple():
  #initial_params = [4., 3., 2., 1000., 1e-3, 10e-3] # With these barely, controllable test fails
  initial_params = [100, 1., 2., 3., 1000., 1e-2, 1e-2]
  dyn = LoudspeakerDynamics(initial_params, outputs=[1])
  x0 = jnp.array([0., 0., 0.]).T
  return dyn, x0

def modelSimpleLinear():
  #initial_params = [4., 3., 2., 1000., 1e-3, 10e-3] # With these barely, controllable test fails
  initial_params = [0, 1., 2., 3., 1000., 1e-2, 1e-2]
  dyn = LoudspeakerDynamics(initial_params, outputs=[1])
  x0 = jnp.array([0., 0., 0.]).T
  return dyn, x0

def modelFull():
  initial_params = [[-1189776480.1292634,
                      542324.9046569373,
                      -1814.696501944539,
                      -32.57636719328277,
                      3.9543755694514506],
                    2.6853699620392533,
                    0.9610089848858704,
                    [172947781225.6865,
                      1257177108.1931014,
                      15038371.688006638,
                      5591.618928926653,
                      747.935417051252],
                    [-18425.153427798694,
                      22.668885453057328,
                      0.23357205756858962,
                      -0.003998867861047569,
                      0.00011528306720962486],
                    0.007127283466160302,
                    0.0002485471030768463,
                    3.4701557050134912,
                    [0.0012384316638823197, 0.00046830900599724033]]
  dyn = LoudspeakerDynamicsFull(initial_params, outputs=[1])
  x0 = jnp.array([0., 0., 0., 0.]).T
  return dyn, x0

def modelFullLinear():
  initial_params = [3.9543755694514506,
                    2.6853699620392533,
                    0.9610089848858704,
                    747.935417051252,
                    0.00011528306720962486,
                    0.007127283466160302,
                    0.0002485471030768463,
                    3.4701557050134912,
                    0.]
  dyn = LoudspeakerDynamicsFull(initial_params, outputs=[1])
  x0 = jnp.array([0., 0., 0., 0.]).T
  return dyn, x0

def linearize(dyn, x0, sr):
  true_model = dx.ForwardModel(dyn, sr)

def main():
  np.random.seed(1)
  # input
  n = 10000
  sr = 96000
  t = jnp.array(np.arange(n)/sr)
  u = 100*jnp.array(np.random.normal(size=n))
  coeffs = dfx.backward_hermite_coefficients(t, u)
  cubic = dfx.CubicInterpolation(t, coeffs)
  ufun = lambda t: cubic.evaluate(t)

  plt.figure(figsize=(8, 4.5))

  # original nonlinear dynamics
  dyn, x0 = modelSimple()
  original_forward = dx.ForwardModel(dyn, sr)
  y = original_forward(t, x0, ufun)[0]
  plt.plot(t, y, label="original system", linewidth=3, alpha=0.5)
  assert not np.isnan(y).any()

  # manual linear dynamics
  lindyn, x0 = modelSimpleLinear()
  original_forward = dx.ForwardModel(lindyn, sr)
  y = original_forward(t, x0, ufun)[0]
  plt.plot(t, y, label="true linearized system", linewidth=3, alpha=0.5)

  # automatically linearized dynamics
  compensator, linearized_dyn = dyn.feedback_linearize(x0)
  linear_estimator = dx.ForwardModel(linearized_dyn, sr)
  y_linear_auto, _ = linear_estimator(t, x0, ufun)
  plt.plot(t, y_linear_auto, label="automatically linearized system")

  # feedback linearized dynamics
  _, x_linear_est = linear_estimator(t, x0, ufun, dense=True)
  def vector_field(t, x, _):
    """Wraps vector field of nonlinear system in compensator."""
    u = ufun(t)
    xest = x_linear_est.evaluate(t)
    v = compensator(u, x, xest)
    return dyn.vector_field(x, v, t)
  sol = dfx.diffeqsolve(dfx.ODETerm(vector_field),
                        dfx.Dopri5(), t0=t[0], t1=t[-1], dt0=1/sr,
                        y0=x0, saveat=dfx.SaveAt(ts=t), max_steps=100*len(t),
                        #stepsize_controller=dfx.ConstantStepSize()
                        )
  y_feedback_lin = jax.vmap(dyn.output)(sol.ys)
  plt.plot(t, y_feedback_lin, '--', label="feedback linearized system")

  # feedback linearized dynamics with linear estimator
  _, x_linear_est = linear_estimator(t, x0, ufun, dense=True)
  def vector_field(t, x, _):
    """Wraps vector field of nonlinear system in compensator."""
    u = ufun(t)
    #xest = x_linear_est.evaluate(t)
    v = compensator(u, x, x)
    return dyn.vector_field(x, v, t)
  # solve compensated system
  term = dfx.ODETerm(vector_field)
  saveat = dfx.SaveAt(ts=t)
  sol = dfx.diffeqsolve(term, dfx.Dopri5(), t0=t[0], t1=t[-1], dt0=1/sr,
                            y0=x0, saveat=saveat, max_steps=100*len(t),
                            #stepsize_controller=dfx.ConstantStepSize()
                            )
  y_feedback_lin_est = jax.vmap(dyn.output)(sol.ys)
  plt.plot(t, y_feedback_lin_est, '--', label="feedback linearized system with linear state estimator")

  plt.ylabel("Displacement [m]")
  plt.xlabel("Time [s]")
  plt.legend()
  plt.savefig("/home/fhchl/Sync/Dendron/vault/assets/images/project.postdoc.lsmod.meet.2022.06.21.weekly.comp_feedback_lin.pdf", bbox_inches=None)
  plt.show()

# Linearized system and feedbacklinearized system not the same
# Next: test with much simpler system, e.g. forced nonlinear harmonic oscillator


main()
