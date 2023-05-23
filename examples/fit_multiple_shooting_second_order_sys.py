"""Example: fit a second-order nonlinear system to data."""

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from dynax import ControlAffine, fit_multiple_shooting, Flow


def tree_pformat(tree):
    return eqx.tree_pformat(tree, short_arrays=False)


# Define a dynamical system of the form
#
# ẋ = f(x) + g(x)u
# y = h(x)
#
# The `ControlAffine` class inherits from eqinox.Module which inherits from
# `dataclasses.dataclass`.
class NonlinearDrag(ControlAffine):
    """Spring-mass-damper system with nonliner drag.

    .. math:: m ẍ +  r ẋ + r2 ẋ |ẋ| + k x = u
              y = x

    """

    # Declare parameters as dataclass fields.
    m: float
    r: float
    r2: float
    k: float

    # Set the number of states (order of system), the number of in- and outputs.
    n_states = 2
    n_inputs = 1
    n_outputs = 1

    # Define the dynamical system via the methods f, g, and h
    def f(self, x, u=None, t=None):
        x1, x2 = x
        return jnp.array(
            [x2, (-self.r * x2 - self.r2 * jnp.abs(x2) * x2 - self.k * x1) / self.m]
        )

    def g(self, x, u=None, t=None):
        return jnp.array([0.0, 1.0 / self.m])

    def h(self, x, u=None, t=None):
        return x[0]


# initiate a dynamical system representing the some "true" parameters
true_system = NonlinearDrag(m=1.0, r=2.0, r2=0.1, k=4.0)
# combine ODE system with ODE solver (Dopri5 and constant stepsize by default)
true_model = Flow(true_system)
print("true system:", true_system)

# some training data using the true model. This could be your measurement data.
t_train = np.linspace(0, 10, 1000)
samplerate = 1 / t_train[1]
np.random.seed(42)
u_train = np.random.normal(size=len(t_train))
initial_x = [0.0, 0.0]
x_train, y_train = true_model(initial_x, t_train, u_train)

# create our model system with some initial parameters
initial_sys = NonlinearDrag(m=1.0, r=1.0, r2=1.0, k=1.0)
print("initial system:", tree_pformat(initial_sys))

# Combine the ODE with an ODE solver
init_model = Flow(initial_sys)
# Fit all parameters with multiple shooting
num_shots = 3
model, x0s, ts, ts0, us = fit_multiple_shooting(
    model=init_model,
    t=t_train,
    y=y_train,
    x0=initial_x,
    u=u_train,
    verbose=0,
    num_shots=num_shots,
)
print("fitted system:", tree_pformat(model.system))

# check the results
x_pred, y_pred = model(initial_x, t_train, u_train)
assert np.allclose(x_train, x_pred, atol=1e-5, rtol=1e-5)

# plot
xs_pred, _ = jax.vmap(model)(x0s, ts0, us)
plt.plot(t_train, x_train, "k--", label="target")
for i in range(num_shots):
    plt.plot(ts[i], xs_pred[i], label="multiple shooting", color=f"C{i}")
    for j in range(x0s.shape[1]):
        plt.scatter(ts[i, 0], x0s[i, j], c=f"C{i}")
plt.plot()
plt.legend()
plt.show()
