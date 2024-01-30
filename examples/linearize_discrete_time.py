import matplotlib.pyplot as plt

from dynax import (
    DynamicalSystem,
    Map,
    discrete_relative_degree,
    DiscreteLinearizingSystem,
    LinearSystem,
)
from equinox.nn import GRUCell
import jax
import jax.numpy as jnp
import numpy as np
from jax.random import PRNGKey


# A nonlinear discrete-time system.
class Recurrent(DynamicalSystem):
    cell: GRUCell

    n_inputs = "scalar"

    def __init__(self, hidden_size, *, key):
        input_size = 1
        self.cell = GRUCell(input_size, hidden_size, use_bias=False, key=key)
        self.n_states = hidden_size

    def vector_field(self, x, u, t=None):
        return self.cell(jnp.array([u]), x)

    def output(self, x, u=None, t=None):
        return x[0]


# A linear reference system.
reference_system = LinearSystem(
    A=jnp.array([[-0.3, 0.1], [0, -0.3]]),
    B=jnp.array([0.0, 1.0]),
    C=jnp.array([1, 0]),
    D=jnp.array(0),
)

# System to contol.
hidden_size = 3
system = Recurrent(hidden_size=hidden_size, key=PRNGKey(0))

# We want the nonlinear systems output to be equal to the reference system's output
# when driven with this input.
inputs = 0.1 * jnp.concatenate((jnp.array([0.1, 0.2, 0.3]), jnp.zeros(10)))

# The relative degree of the reference system can be larger or equal to the relative
# degree of the nonlinear system. Here we test for the relative degree with a set of
# points and inputs.
reldeg = discrete_relative_degree(
    system, np.random.normal(size=(inputs.size, system.n_states)), inputs
)
print("Relative degree of nonlinear system:", reldeg)
print(
    "Relative degree of reference system:",
    discrete_relative_degree(
        reference_system,
        np.random.normal(size=(inputs.size, reference_system.n_states)),
        inputs,
    ),
)

# We compute the input signal that forces the outputs of the nonlinear and reference
# systems to be equal by solving a coupled system.
linearizing_system = DiscreteLinearizingSystem(system, reference_system, reldeg)

# The output of this system when driven with the reference input is the linearizing
# input. The coupled system as an extra state used internally.
_, linearizing_inputs = Map(linearizing_system)(
    jnp.zeros(system.n_states + reference_system.n_states + 1), u=inputs
)

# Lets simulate the original system, the linear reference and the linearized system.
states_orig, output_orig = Map(system)(x0=jnp.zeros(hidden_size), u=inputs)
_, output_ref = Map(reference_system)(x0=jnp.zeros(reference_system.n_states), u=inputs)
_, output_linearized = Map(system)(jnp.zeros(hidden_size), u=linearizing_inputs)

assert np.allclose(output_ref, output_linearized)

plt.plot(output_orig, label="GRUCell")
plt.plot(output_ref, label="linear reference")
plt.plot(output_linearized, "--", label="input-output linearized GRU")
plt.legend()
plt.figure()
plt.plot(linearizing_inputs, label="linearizing input")
plt.legend()
plt.show()


