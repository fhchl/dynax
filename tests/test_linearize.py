import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from dynax import (
    ControlAffine,
    DynamicalSystem,
    DynamicStateFeedbackSystem,
    Flow,
    LinearSystem,
)
from dynax.example_models import NonlinearDrag, Sastry9_9
from dynax.linearize import input_output_linearize, is_controllable, relative_degree


tols = dict(rtol=1e-04, atol=1e-06)


class SpringMassDamperWithOutput(ControlAffine):
    m = 0.1
    r = 0.1
    k = 0.1
    out: int
    n_states = 2
    n_inputs = 1

    def f(self, x):
        x1, x2 = x
        return jnp.array([x2, (-self.r * x2 - self.k * x1) / self.m])

    def g(self, x):
        return jnp.array([0, 1 / self.m])

    def h(self, x):
        return x[np.array(self.out)]


def test_relative_degree():
    xs = np.random.normal(size=(100, 2))
    # output is position
    sys = SpringMassDamperWithOutput(out=0)
    assert relative_degree(sys, xs) == 2
    # output is velocity
    sys = SpringMassDamperWithOutput(out=1)
    assert relative_degree(sys, xs) == 1


def test_input_output_linearize_single_output():
    """Feedback linearized system equals system linearized around x0."""
    sys = NonlinearDrag(0.1, 0.1, 0.1, 0.1)
    ref = sys.linearize()
    xs = np.random.normal(size=(100, sys.n_states))
    reldeg = relative_degree(sys, xs)
    feedbacklaw = input_output_linearize(sys, reldeg, ref)
    feedback_sys = DynamicStateFeedbackSystem(sys, ref, feedbacklaw)
    t = np.linspace(0, 1)
    u = np.sin(t)
    npt.assert_allclose(
        Flow(ref)(np.zeros(sys.n_states), t, u)[1],
        Flow(feedback_sys)(np.zeros(feedback_sys.n_states), t, u)[1],
        **tols,
    )


def test_input_output_linearize_multiple_outputs():
    """Can select an output for linearization."""
    sys = SpringMassDamperWithOutput(out=[0, 1])
    ref = sys.linearize()
    for out_idx in range(2):
        out_idx = 1
        xs = np.random.normal(size=(100, sys.n_states))
        reldeg = relative_degree(sys, xs, output=out_idx)
        feedbacklaw = input_output_linearize(sys, reldeg, ref, output=out_idx)
        feedback_sys = DynamicStateFeedbackSystem(sys, ref, feedbacklaw)
        t = np.linspace(0, 1)
        u = np.sin(t) * 0.1
        y_ref = Flow(ref)(np.zeros(sys.n_states), t, u)[1]
        y = Flow(feedback_sys)(np.zeros(feedback_sys.n_states), t, u)[1]
        npt.assert_allclose(y_ref[:, out_idx], y[:, out_idx], **tols)
