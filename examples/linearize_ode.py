import matplotlib.pyplot as plt
import numpy as np

from dynax import (
    Flow,
    LinearizingSystem,
    relative_degree,
)
from dynax.example_models import NonlinearDrag


# The system to control: a simple spring-mass-damper system with strong nonlinear drag.
system = NonlinearDrag(r=1.0, r2=5.0, k=1.0, m=1.0)

# The linear reference system is the system linearized around the origin.
reference_system = system.linearize()

# We want the nonlinear systems output to be equal to the reference system's output
# when driven with the following input.
t = np.linspace(0, 10, 1000)
u = 10 * np.sin(2 * np.pi * t)

# Compute the relative degree of the system over a set of test states.
reldeg = relative_degree(
    sys=system, xs=np.random.normal(size=(100, len(system.initial_state)))
)

# The input signal that forces the outputs of the nonlinear and reference
# systems to be equal is computed by solving a coupled ODE system constructed by
# `dynax.LinearizingSystem`.
linearizing_system = LinearizingSystem(system, reference_system, reldeg)

# The output of this system when driven with the reference input is the linearizing
# input.
_, linearizing_inputs = Flow(linearizing_system)(t=t, u=u)

# Lets simulate the original system,
states_orig, output_orig = Flow(system)(t=t, u=u)
# the linear reference system,
_, output_ref = Flow(reference_system)(t=t, u=u)
# and the nonlinear system driven with the linearizing signal.
_, output_linearized = Flow(system)(t=t, u=linearizing_inputs)


plt.plot(t, output_orig, label="nonlinear drag")
plt.plot(t, output_ref, label="linear reference")
plt.plot(t, output_linearized, "--", label="input-output linearized")
plt.legend()
plt.figure()
plt.plot(t, u, label="input to reference system")
plt.plot(t, linearizing_inputs, label="linearizing input")
plt.legend()
plt.show()

# The output of the linearized system is equal to the output of the reference system!
assert np.allclose(output_ref, output_linearized, atol=1e-4)
