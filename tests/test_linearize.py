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

    def f(self, x, t=None):
        x1, x2 = x
        return jnp.array([x2, (-self.r * x2 - self.k * x1) / self.m])

    def g(self, x, t=None):
        return jnp.array([0, 1 / self.m])

    def h(self, x, t=None):
        return x[np.array(self.out)]


def test_relative_degree():
    xs = np.random.normal(size=(100, 2))
    # output is position
    sys = SpringMassDamperWithOutput(out=0)
    assert relative_degree(sys, xs) == 2
    # output is velocity
    sys = SpringMassDamperWithOutput(out=1)
    assert relative_degree(sys, xs) == 1


def test_is_controllable():
    n = 3
    A = np.diag(np.arange(n))
    B = np.ones((n, 1))
    assert is_controllable(A, B)

    A[1, :] = A[0, :]
    assert not is_controllable(A, B)


def test_linearize_lin2lin():
    n, m, p = 3, 2, 1
    A = np.random.normal(size=(n, n))
    B = np.random.normal(size=(n, m))
    C = np.random.normal(size=(p, n))
    D = np.random.normal(size=(p, m))
    sys = LinearSystem(A, B, C, D)
    linsys = sys.linearize()
    assert np.allclose(A, linsys.A)
    assert np.allclose(B, linsys.B)
    assert np.allclose(C, linsys.C)
    assert np.allclose(D, linsys.D)


def test_linearize_dyn2lin():
    class TestSys(DynamicalSystem):
        n_states = 1
        n_inputs = 1
        vector_field = lambda self, x, u=None, t=None: -1 * x + 2 * u
        output = lambda self, x, u=None, t=None: 3 * x + 4 * u

    sys = TestSys()
    linsys = sys.linearize()
    assert np.array_equal(linsys.A, [[-1.0]])
    assert np.array_equal(linsys.B, [[2.0]])
    assert np.array_equal(linsys.C, [[3.0]])
    assert np.array_equal(linsys.D, [[4.0]])


def test_linearize_sastry9_9():
    """Linearize should return 2d-arrays. Refererence computed by hand."""
    sys = Sastry9_9()
    linsys = sys.linearize()
    assert np.array_equal(linsys.A, [[0, 0, 0], [1, 0, 0], [1, -1, 0]])
    assert np.array_equal(linsys.B, [[1], [1], [0]])
    assert np.array_equal(linsys.C, [[0, 0, 1]])
    assert np.array_equal(linsys.D, [[0.0]])


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
