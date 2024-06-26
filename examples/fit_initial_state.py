"""Example: fit a second-order nonlinear system to data."""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from dynax import (
    AbstractControlAffine,
    fit_csd_matching,
    fit_least_squares,
    Flow,
    free_field,
)


# Define a dynamical system of the form
#
# ẋ = f(x) + g(x)u
# y = h(x)
#
# The `ControlAffine` class inherits from eqinox.Module which inherits from
# `dataclasses.dataclass`.
class NonlinearDrag(AbstractControlAffine):
    """Spring-mass-damper system with nonliner drag.

    .. math:: m ẍ +  r ẋ + r2 ẋ |ẋ| + k x = u
              y = x

    """

    # Declare parameters as dataclass fields.
    m: float
    r: float
    r2: float
    k: float

    # The initial_state attribute is static by default. If we want to make it learnable
    # we must declare it using the `free_field` function.
    initial_state: jnp.ndarray = free_field(init=True)

    n_inputs = "scalar"

    # Define the dynamical system via the methods f, g, and h
    def f(self, x):
        x1, x2 = x
        return jnp.array(
            [x2, (-self.r * x2 - self.r2 * jnp.abs(x2) * x2 - self.k * x1) / self.m]
        )

    def g(self, x):
        return jnp.array([0.0, 1.0 / self.m])

    def h(self, x):
        return x[0]


# initiate a dynamical system representing the some "true" parameters
true_system = NonlinearDrag(
    m=1.0, r=2.0, r2=0.1, k=4.0, initial_state=jnp.array([1.0, 1.0])
)
# combine ODE system with ODE solver (Dopri5 and constant stepsize by default)
true_model = Flow(true_system)
print("true system:", true_system)

# some training data using the true model. This could be your measurement data.
t_train = np.linspace(0, 10, 1000)
samplerate = 1 / t_train[1]
np.random.seed(42)
u_train = np.random.normal(size=len(t_train))
x_train, y_train = true_model(t_train, u_train)

# create our model system with some initial parameters
initial_sys = NonlinearDrag(
    m=1.0, r=1.0, r2=1.0, k=1.0, initial_state=jnp.array([0.0, 0.0])
)
print("initial system:", initial_sys)

# If we have long-duration, wide-band input data we can fit the linear
# parameters by matching the transfer-functions. In this example the result is
# not very good.
initial_sys = fit_csd_matching(
    initial_sys, u_train, y_train, samplerate, nperseg=100
).result
print("linear params fitted:", initial_sys)

# Combine the ODE with an ODE solver
init_model = Flow(initial_sys)
# Fit all parameters with previously estimated parameters as a starting guess.
pred_model = fit_least_squares(
    model=init_model, t=t_train, y=y_train, u=u_train, verbose=0
).result
print("fitted system:", pred_model.system)

# check the results
x_pred, y_pred = pred_model(t_train, u_train)
assert np.allclose(x_train, x_pred)

plt.plot(t_train, x_train, label="target")
plt.plot(t_train, x_pred, "--", label="prediction")
plt.legend()
plt.show()
