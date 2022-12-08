import jax
import numpy as np
import matplotlib.pyplot as plt
import diffrax as dfx

from dynax.linearize import input_output_linearize
from dynax import ForwardModel
from dynax.models import NonlinearDrag

jax.config.update("jax_enable_x64", True)

# a nonlinear drag model
sys = NonlinearDrag(0.1, 0.1, 0.1, 0.1)
model = ForwardModel(sys)
t = np.linspace(0, 50, 1000)
u = np.sin(0.1*2*np.pi*t)
x0 = [0., 0.]
x, _ = model(x0, t, u)

# linearize model around x=0
linsys = sys.linearize()
linmodel = ForwardModel(linsys, step=dfx.PIDController(rtol=1e-4, atol=1e-6))
x_lin, _ = linmodel(x0, t, u)

# input-output linearized model
feedbacklaw = input_output_linearize(sys, reldeg=2, reference=linsys)
u_comp = jax.vmap(feedbacklaw)(x_lin, u)
x_comp, y = model(x0, t, u_comp)

# plot states
plt.plot(t, x, '-')
plt.plot(t, x_lin, '-', linewidth=5, alpha=0.5)
plt.plot(t, x_comp, ':')
plt.legend([
  "nonlin $x$", "nonlin $\dot x$",
  "linearized $x$", "linearized $\dot x$",
  "compensated $x$", "compensated $\dot x$",
]);
plt.xlabel("Time")
plt.ylabel("States");

# plot inputs
plt.figure()
plt.plot(t, u)
plt.plot(t, u_comp)
plt.legend(["original input", "linearizing input"])
plt.xlabel("Time")
plt.ylabel("Input")

plt.show()