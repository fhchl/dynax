import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from equinox.nn import GRUCell
from jax.random import PRNGKey

from dynax import (
    AbstractSystem,
    discrete_relative_degree,
    DiscreteLinearizingSystem,
    LinearSystem,
    Map,
)


# The system to control: a simple RNN with a GRU cell
class Recurrent(AbstractSystem):
    cell: GRUCell

    n_inputs = "scalar"

    def __init__(self, hidden_size, *, key):
        self.cell = GRUCell(
            input_size=1, hidden_size=hidden_size, use_bias=False, key=key
        )
        self.initial_state = np.zeros(hidden_size)

    def vector_field(self, x, u, t=None):
        return self.cell(jnp.array([u]), x)

    def output(self, x, u=None, t=None):
        return x[0]


hidden_size = 3
system = Recurrent(hidden_size=hidden_size, key=PRNGKey(0))

# A linear reference system.
reference_system = LinearSystem(
    A=jnp.array([[-0.3, 0.1], [0, -0.3]]),
    B=jnp.array([0.0, 1.0]),
    C=jnp.array([1, 0]),
    D=jnp.array(0),
)

# We want the nonlinear systems output to be equal to the reference system's output
# when driven with the following input.
u = 0.1 * jnp.concatenate((jnp.array([0.1, 0.2, 0.3]), jnp.zeros(10)))

# The relative degree of the reference system can be larger or equal to the relative
# degree of the nonlinear system. Here we test for the relative degree with a set of
# points and inputs.
reldeg = discrete_relative_degree(
    system, np.random.normal(size=(len(u),) + system.initial_state.shape), u
)
print("Relative degree of nonlinear system:", reldeg)
print(
    "Relative degree of reference system:",
    discrete_relative_degree(
        reference_system,
        np.random.normal(size=(len(u),) + reference_system.initial_state.shape),
        u,
    ),
)

# We compute the input signal that forces the outputs of the nonlinear and reference
# systems to be equal by solving a coupled ODE system that is constructed by
# `dynax.DiscreteLinearizingSystem`
linearizing_system = DiscreteLinearizingSystem(system, reference_system, reldeg)

# The output of this system when driven with the reference input is the linearizing
# input.
_, linearizing_inputs = Map(linearizing_system)(u=u)

# Lets simulate the original system,
states_orig, output_orig = Map(system)(u=u)
# the linear reference system,
_, output_ref = Map(reference_system)(u=u)
# and the nonlinear system driven with the linearizing signal.
_, output_linearized = Map(system)(u=linearizing_inputs)

# The output of the linearized system is equal to the output of the reference system!
assert np.allclose(output_ref, output_linearized)

plt.plot(output_orig, label="GRUCell")
plt.plot(output_ref, label="linear reference")
plt.plot(output_linearized, "--", label="input-output linearized GRU")
plt.legend()
plt.figure()
plt.plot(u, label="input to reference system")
plt.plot(linearizing_inputs, label="linearizing input")
plt.legend()
plt.show()
