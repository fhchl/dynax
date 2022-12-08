import diffrax as dfx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from dynax import (ControlAffine, DynamicStateFeedbackSystem, ForwardModel,
                   LinearSystem)
from dynax.ad import lie_derivative
from dynax.linearize import input_output_linearize, relative_degree
from dynax.models import (NonlinearDrag, PolyNonLinLS,
                          PolyNonLinSLSL2R2GenCunLiDyn)

jax.config.update("jax_enable_x64", True)


def linearizing_diffeomorphism(sys, reldeg):
  def diffeo(x):
    lieds = [lie_derivative(sys.f, sys.y, n) for n in range(reldeg-1)]
    return jnp.array([lied(x) for lied in lieds])
  return diffeo

# input
sr = 96000
duration = 1
t = np.linspace(0, duration, sr)
u = 5*np.sin(20*2*np.pi*t) + 5 * np.sin(200*2*np.pi*t)

import scipy.signal as sig
np.random.seed(0)
bandpass = np.asarray([10, 100])
sos = sig.butter(8, bandpass / (sr / 2), "bandpass", output='sos')
u = np.random.normal(0, 1, sr*duration)
u = sig.sosfilt(sos, u)
u = u / np.std(u) * 8


# nonlinear system to be linearized
sys = PolyNonLinLS(out=[1])
sys = NonlinearDrag(0.1, 0.1, 0.1, 0.1)
sys = PolyNonLinSLSL2R2GenCunLiDyn(out=[1])
init_state = np.zeros(sys.n_states)

# forward model = ODE + solver
solver_opt = dict(solver=dfx.Kvaerno5(), step=dfx.PIDController(rtol=1e-8, atol=1e-12))
model = ForwardModel(sys, **solver_opt)

# uncompensated response
x, y = model(init_state, t, u)

# linearize model around x=0
linsys = sys.linearize()
linmodel = ForwardModel(linsys, **solver_opt)
z, y_m = linmodel(init_state, t, u)

# input-output linearized model
reldeg = relative_degree(sys, x, 5)
feedbacklaw = input_output_linearize(sys, reldeg, ref=linsys)
feedbacksys = DynamicStateFeedbackSystem(sys, linsys, feedbacklaw)
feedbackmodel = ForwardModel(feedbacksys, **solver_opt)
xz, y_comp = feedbackmodel(np.concatenate((init_state, init_state)), t, u,
                           max_steps=1000000)

x_comp, z = xz[:, :sys.n_states], xz[:, sys.n_states:]
u_comp = jax.vmap(feedbacklaw)(x_comp, z, u)

def nrmse(target, prediction, axis=0):
  return np.sqrt(np.mean((target-prediction)**2, axis=axis))

print("NRMSE y-y_m:", nrmse(y_comp, y_m))


# plot nonlinearities
fig, ax = plt.subplots(nrows=4)
i, d, v, i2, v2 = x.T
ax[0].plot(d, sys._Bl(d))
ax[1].plot(d, sys._K(d))
ax[2].plot(d, sys._L(d, 0))
ax[3].plot(i, sys._L(0, i))

# fig, ax = plt.subplots(nrows=3)
# i, d, v = x.T
# ax[0].plot(d, sys._Bl(d))
# ax[1].plot(d, sys._K(d))
# ax[2].plot(d, sys._L(d))

# plot inputs
plt.figure()
plt.plot(t, u)
plt.plot(t, u_comp)
plt.legend(["original input", "$u(x, z, r)$"])
plt.xlabel("Time")
plt.ylabel("Input")

# plot states
fig, axes = plt.subplots(nrows=sys.n_states)
state_names = ["$i$", "$x$", "$\dot x$", "$i_2$"," $v_2$"]
for i in range(sys.n_states):
  axes[i].plot(t, x.T[i], '-', label=f"nonlin {state_names[i]}")
  axes[i].plot(t, z.T[i], '-', label=f"linear {state_names[i]}", linewidth=5, alpha=0.5)
  axes[i].plot(t, x_comp.T[i], '-', label=f"compensated {state_names[i]} with $u(x, z, r)$")
  axes[i].legend()
  axes[i].set_xlabel("Time")
  axes[i].set_ylabel("States");

plt.show()


"""
Learnings:

Even though relative degree == number of states, we don't have a full-state
feedback linearization. Thus, we need to use u(x, z, r) style of feedback.

Above approach needs VERY high sample rate, precision in adaptive step size.
"""