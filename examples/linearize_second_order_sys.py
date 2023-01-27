import diffrax as dfx
import jax
import matplotlib.pyplot as plt
import numpy as np
from dynax import DynamicStateFeedbackSystem, Flow
from dynax.example_models import NonlinearDrag
from dynax.linearize import input_output_linearize, relative_degree


jax.config.update("jax_enable_x64", True)
solver_opt = dict(solver=dfx.Kvaerno5(), step=dfx.PIDController(rtol=1e-5, atol=1e-7))

# a nonlinear drag model
sys = NonlinearDrag(1, 1, 0.2, 1)
model = Flow(sys, **solver_opt)
t = np.linspace(0, 50, 1000)
u = 10 * np.sin(0.1 * 2 * np.pi * t)
x0 = [0.0, 0.0]
x, _ = model(x0, t, u)
reldeg = relative_degree(sys, x)

# linearize model around x=0
refsys = sys.linearize()
refmodel = Flow(refsys, **solver_opt)
x_ref, _ = refmodel(x0, t, u)

# input-output linearized model
feedbacklaw = input_output_linearize(sys, reldeg, ref=refsys)
feedback_sys = DynamicStateFeedbackSystem(sys, refsys, feedbacklaw)
feedback_model = Flow(feedback_sys, **solver_opt)
xz_comp, _ = feedback_model(np.zeros(feedback_sys.n_states), t, u)
x_comp = xz_comp[:, : sys.n_states]
u_comp = jax.vmap(feedbacklaw)(x_comp, x_ref, u)

assert np.allclose(x_ref, x_comp, atol=1e-3, rtol=1e-3)

# plot states
plt.plot(t, x, "-")
plt.plot(t, x_ref, "-", linewidth=5, alpha=0.5)
plt.plot(t, x_comp, ":")
plt.legend(
    [
        "nonlin $x$",
        r"nonlin $\dot x$",
        "linearized $x$",
        r"linearized $\dot x$",
        "compensated $x$",
        r"compensated $\dot x$",
    ]
)
plt.xlabel("Time")
plt.ylabel("States")

# plot inputs
plt.figure()
plt.plot(t, u)
plt.plot(t, u_comp)
plt.legend(["original input", "linearizing input"])
plt.xlabel("Time")
plt.ylabel("Input")

plt.show()
