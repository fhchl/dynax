"""Fit a parameters and initial values of an ODE."""

import jax
import matplotlib.pyplot as plt
import numpy as np

from dynax import field, fit_least_squares, Flow, pretty
from dynax.example_models import NonlinearDrag


# Declare the initial values as trainable using the `field` function.
class NonlinearDragWithInitialValues(NonlinearDrag):
    initial_state: jax.Array = field(init=True)


# Initiate a dynamical system representing the some "true" parameters.
true_system = NonlinearDragWithInitialValues(
    m=1.0, r=2.0, r2=0.1, k=4.0, initial_state=(0.5, 1.0)
)

# Combine ODE system and ODE solver (Dopri5 and constant stepsize by default).
true_model = Flow(true_system)
print("true system:", pretty(true_system))

# Create some training data using the true model. This could be your measurement data.
t_train = np.linspace(0, 2, 100)
samplerate = 1 / t_train[1]
np.random.seed(42)
u_train = np.random.normal(size=len(t_train))
x_train, y_train = true_model(t_train, u_train)

# Create our model system with some initial parameters.
initial_sys = NonlinearDragWithInitialValues(
    m=1.0, r=1.0, r2=1.0, k=1.0, initial_state=(0.0, 0.0)
)
print("initial system:", pretty(initial_sys))

# Combine the ODE with an ODE solver.
init_model = Flow(initial_sys)

# Fit the parameters of the nonlinear system including the initial state.
pred_model = fit_least_squares(
    model=init_model, t=t_train, y=y_train, u=u_train, verbose=0
).result
print("fitted system:", pretty(pred_model.system))

# Check the results.
x_pred, y_pred = pred_model(t_train, u_train)
assert np.allclose(x_train, x_pred)
assert np.allclose(true_system.initial_state, pred_model.system.initial_state)

plt.plot(t_train, x_train, "--", label="target")
plt.plot(t_train, x_pred, label="prediction")
plt.legend()
plt.show()
