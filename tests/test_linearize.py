import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from dynax import (
    AbstractControlAffine,
    AbstractSystem,
    discrete_relative_degree,
    DiscreteLinearizingSystem,
    DynamicStateFeedbackSystem,
    Flow,
    input_output_linearize,
    LinearSystem,
    Map,
    relative_degree,
)
from dynax.custom_types import Array, FloatScalarLike
from dynax.example_models import NonlinearDrag, Sastry9_9
from dynax.linearize import (
    is_controllable,
)
from dynax.system import static_field


class Allpass(AbstractControlAffine):
    initial_state: Array = static_field(default_factory=lambda: jnp.zeros(0))
    n_inputs: int | str = static_field(default="scalar")

    def f(self, x: Array) -> Array:
        return jnp.array(0.0)

    def g(self, x: Array) -> Array:
        return jnp.array(0.0)

    def h(self, x: Array) -> Array:
        return jnp.array(0.0)

    def i(self, x: Array) -> Array:
        return jnp.array(1.0)


class SpringMassDamperWithOutput(AbstractControlAffine):
    m: float = 0.1
    r: float = 0.1
    k: float = 0.1
    out: int = 0

    initial_state: Array = static_field(default_factory=lambda: jnp.zeros(2))
    n_inputs: int | str = static_field(default="scalar")

    def f(self, x: Array) -> Array:
        x1, x2 = x
        return jnp.array([x2, (-self.r * x2 - self.k * x1) / self.m])

    def g(self, x: Array) -> Array:
        return jnp.array([0, 1 / self.m])

    def h(self, x: Array) -> Array:
        return x[np.array(self.out)]


def test_relative_degree():
    xs = jnp.asarray(np.random.normal(size=(100, 2)))
    # output is position
    sys = SpringMassDamperWithOutput(out=0)
    assert relative_degree(sys, xs) == 2
    # output is velocity
    sys = SpringMassDamperWithOutput(out=1)
    assert relative_degree(sys, xs) == 1

    xs = jnp.asarray(np.random.normal(size=100))
    assert relative_degree(Allpass(), xs) == 0


def test_discrete_relative_degree():
    xs = jnp.asarray(np.random.normal(size=(100, 2)))
    us = jnp.asarray(np.random.normal(size=(100)))

    sys = SpringMassDamperWithOutput(out=0)
    assert discrete_relative_degree(sys, xs, us) == 2

    sys = SpringMassDamperWithOutput(out=1)
    assert discrete_relative_degree(sys, xs, us) == 1

    xs = jnp.asarray(np.random.normal(size=100))
    assert discrete_relative_degree(Allpass(), xs, us) == 0


def test_is_controllable():
    n = 3
    A = np.diag(np.arange(n))
    B = np.ones((n, 1))
    assert is_controllable(A, B)

    A[1, :] = A[0, :]
    assert not is_controllable(A, B)


def test_linearize_lin2lin():
    n, m, p = 3, 2, 1
    A = jnp.array(np.random.normal(size=(n, n)))
    B = jnp.array(np.random.normal(size=(n, m)))
    C = jnp.array(np.random.normal(size=(p, n)))
    D = jnp.array(np.random.normal(size=(p, m)))
    sys = LinearSystem(A, B, C, D)  # type: ignore[call-arg]
    linsys = sys.linearize()
    assert np.allclose(A, linsys.A)
    assert np.allclose(B, linsys.B)
    assert np.allclose(C, linsys.C)
    assert np.allclose(D, linsys.D)


def test_linearize_dyn2lin():
    class ScalarScalar(AbstractSystem):
        initial_state: Array = static_field(default_factory=lambda: jnp.array(0.0))
        n_inputs: int | str = static_field(default="scalar")

        def vector_field(
            self, x: Array, u: Array | None = None, t: FloatScalarLike | None = None
        ) -> Array:
            return -1 * x + 2 * (u if u is not None else jnp.zeros(()))

        def output(
            self, x: Array, u: Array | None = None, t: FloatScalarLike | None = None
        ) -> Array:
            return 3 * x + 4 * (u if u is not None else jnp.zeros(()))

    sys = ScalarScalar()
    linsys = sys.linearize()
    assert np.array_equal(linsys.A, -1.0)
    assert np.array_equal(linsys.B, 2.0)
    assert np.array_equal(linsys.C, 3.0)
    assert np.array_equal(linsys.D, 4.0)


def test_linearize_sastry9_9():
    """Linearize should return 2d-arrays. Refererence computed by hand."""
    sys = Sastry9_9()
    linsys = sys.linearize()
    assert np.array_equal(linsys.A, [[0, 0, 0], [1, 0, 0], [1, -1, 0]])
    assert np.array_equal(linsys.B, [1, 1, 0])
    assert np.array_equal(linsys.C, [0, 0, 1])
    assert np.array_equal(linsys.D, 0.0)


def test_input_output_linearize_single_output():
    """Feedback linearized system equals system linearized around x0."""
    sys = NonlinearDrag(jnp.array(0.1), jnp.array(0.1), jnp.array(0.1), jnp.array(0.1))
    ref = sys.linearize()
    xs = jnp.asarray(np.random.normal(size=(100,) + sys.initial_state.shape))
    reldeg = relative_degree(sys, xs)
    feedbacklaw = input_output_linearize(sys, reldeg, ref)
    feedback_sys = DynamicStateFeedbackSystem(sys, ref, feedbacklaw)
    t = jnp.linspace(0, 0.1)
    u = jnp.sin(t)
    npt.assert_allclose(
        Flow(ref)(t, u)[1],
        Flow(feedback_sys)(t, u)[1],
        rtol=1e-04,
        atol=1e-06,
    )


def test_input_output_linearize_multiple_outputs():
    """Can select an output for linearization."""
    sys = SpringMassDamperWithOutput(out=[0, 1])  # type: ignore[arg-type]
    ref = sys.linearize()
    for out_idx in range(2):
        out_idx = 1
        xs = jnp.asarray(np.random.normal(size=(100,) + sys.initial_state.shape))
        reldeg = relative_degree(sys, xs, output=out_idx)
        feedbacklaw = input_output_linearize(sys, reldeg, ref, output=out_idx)
        feedback_sys = DynamicStateFeedbackSystem(sys, ref, feedbacklaw)
        t = jnp.linspace(0, 1)
        u = jnp.sin(t) * 0.1
        y_ref = Flow(ref)(t, u)[1]
        y = Flow(feedback_sys)(t, u)[1]
        npt.assert_allclose(y_ref[:, out_idx], y[:, out_idx], rtol=1e-04, atol=1e-06)


class Lee7_4_5(AbstractSystem):
    initial_state: Array = static_field(default_factory=lambda: jnp.zeros(2))
    n_inputs: int | str = static_field(default="scalar")

    def vector_field(
        self, x: Array, u: Array | None = None, t: FloatScalarLike | None = None
    ) -> Array:
        x1, x2 = x
        u_val = u if u is not None else jnp.zeros(())
        return 0.1 * jnp.array([x1 + x1**3 + x2, x2 + x2**3 + u_val])

    def output(
        self, x: Array, u: Array | None = None, t: FloatScalarLike | None = None
    ) -> Array:
        return x[0]


def test_discrete_input_output_linearize():
    sys = Lee7_4_5()
    refsys = sys.linearize()
    xs = jnp.asarray(np.random.normal(size=(100, 2)))
    us = jnp.asarray(np.random.normal(size=100))
    reldeg = discrete_relative_degree(sys, xs, us)
    assert reldeg == 2

    feedback_sys = DiscreteLinearizingSystem(sys, refsys, reldeg)
    t = jnp.linspace(0, 0.001, 10)
    u = jnp.cos(t) * 0.1
    _, v = Map(feedback_sys)(t, u)
    _, y = Map(sys)(t, u)
    _, y_ref = Map(refsys)(t, u)

    npt.assert_allclose(y_ref, y, rtol=1e-04, atol=1e-06)
