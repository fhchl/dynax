"""Example: fit a second-order nonlinear system to data."""

import equinox as eqx
import jax
import matplotlib.pyplot as plt
import numpy as np

from dynax import fit_multiple_shooting, Flow, pretty
from dynax.example_models import LotkaVolterra


# Initiate a dynamical system representing the some "true" parameters.
true_system = LotkaVolterra(alpha=0.1, beta=0.2, gamma=0.3, delta=0.4)
# Combine ODE system with ODE solver (Dopri5 and constant stepsize by default)
true_model = Flow(true_system)
print("true system:", true_system)

# Generate training data using the true model. This could be your measurement data.
t_train = np.linspace(0, 100, 1000)
_, y_train = true_model(t_train)

# Initiate ODE with some initial parameters.
initial_sys = LotkaVolterra(alpha=0.5, beta=0.5, gamma=0.5, delta=0.5)
print("initial system:", pretty(initial_sys))

# Combine the ODE with an ODE solver.
init_model = Flow(initial_sys)

# Fiting with single shooting fails: the optimizer gets stuck in local minima.
num_shots = 1
res = fit_multiple_shooting(
    model=init_model,
    t=t_train,
    y=y_train,
    verbose=2,
    num_shots=num_shots,
)
model = res.result
x0s = res.x0s
ts = res.ts
ts0 = res.ts0
print("single shooting:", pretty(model.system))

plt.figure()
plt.title("single shooting")
_, ys_pred = jax.vmap(model)(ts0, initial_state=x0s)
plt.plot(t_train, y_train, "k--", label="target")
for i in range(num_shots):
    plt.plot(ts[i], ys_pred[i], label="fitted", color=f"C{i}")
    for j in range(x0s.shape[1]):
        plt.scatter(ts[i, 0], x0s[i, j], c=f"C{i}")
plt.plot()
plt.legend()

# Multiple shooting to the rescue.
num_shots = 3
res = fit_multiple_shooting(
    model=init_model,
    t=t_train,
    y=y_train,
    verbose=2,
    num_shots=num_shots,
)
model = res.result
x0s = res.x0s
ts = res.ts
ts0 = res.ts0
print("multiple shooting:", pretty(model.system))

plt.figure()
plt.title("multiple shooting")
_, ys_pred = jax.vmap(model)(ts0, initial_state=x0s)
plt.plot(t_train, y_train, "k--", label="target")
for i in range(num_shots):
    plt.plot(ts[i], ys_pred[i], label="fitted", color=f"C{i}")
    for j in range(x0s.shape[1]):
        plt.scatter(ts[i, 0], x0s[i, j], c=f"C{i}")
plt.plot()
plt.legend()

plt.show()

# Check the results
_, y_pred = model(t_train)
assert eqx.tree_equal(model.system, true_system, rtol=1e-3, atol=1e-3)
assert np.allclose(y_train, y_pred, atol=1e-5, rtol=1e-5)
