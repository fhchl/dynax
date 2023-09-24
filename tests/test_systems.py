import diffrax as dfx
import jax.numpy as jnp
import pytest

from dynax import FeedbackSystem, Flow, LinearSystem, SeriesSystem


tols = dict(rtol=1e-04, atol=1e-06)


def test_linear_system():
    b = 2
    c = 1  # critical damping as b**2 == 4*c
    uconst = 1

    A = jnp.array([[0, 1], [-c, -b]])
    B = jnp.array([[0], [1]])
    C = jnp.array([[1, 0]])
    D = jnp.zeros((1, 1))
    sys = LinearSystem(A, B, C, D)

    def x(t, x0, uconst):
        """Solution to critically damped linear second-order system."""
        C2 = x0 - uconst / c
        C1 = b / 2 * C2
        return jnp.exp(-b * t / 2) * (C1 * t + C2) + uconst / c

    x0 = jnp.array([1., 0.])  # x(t=0)=1, dx(t=0)=0
    t = jnp.linspace(0, 1)
    u = jnp.ones_like(t) * uconst
    model = Flow(sys, step=dfx.PIDController(rtol=1e-7, atol=1e-9))
    x_pred = model(x0, t, u)[1]
    x_true = x(t, x0[0], uconst)
    assert jnp.allclose(x_true, x_pred)


@pytest.mark.skip
def test_series():
    n1, m1, p1 = 4, 3, 2
    A1 = np.random.randint(-5, 5, size=(n1, n1))
    B1 = np.random.randint(-5, 5, size=(n1, m1))
    C1 = np.random.randint(-5, 5, size=(p1, n1))
    D1 = np.random.randint(-5, 5, size=(p1, m1))
    sys1 = LinearSystem(A1, B1, C1, D1)
    n2, m2, p2 = 5, p1, 3
    A2 = np.random.randint(-5, 5, size=(n2, n2))
    B2 = np.random.randint(-5, 5, size=(n2, m2))
    C2 = np.random.randint(-5, 5, size=(p2, n2))
    D2 = np.random.randint(-5, 5, size=(p2, m2))
    sys2 = LinearSystem(A2, B2, C2, D2)
    sys = SeriesSystem(sys1, sys2)
    linsys = sys.linearize()
    assert np.array_equal(
        linsys.A, np.block([[A1, np.zeros((n1, n2))], [B2.dot(C1), A2]])
    )
    assert np.array_equal(linsys.B, np.block([[B1], [B2.dot(D1)]]))
    assert np.array_equal(linsys.C, np.block([[D2.dot(C1), C2]]))
    assert np.array_equal(linsys.D, D2.dot(D1))


@pytest.mark.skip
def test_feedback():
    n1, m1, p1 = 4, 3, 2
    A1 = np.random.randint(-5, 5, size=(n1, n1))
    B1 = np.random.randint(-5, 5, size=(n1, m1))
    C1 = np.random.randint(-5, 5, size=(p1, n1))
    D1 = np.zeros((p1, m1))
    sys1 = LinearSystem(A1, B1, C1, D1)
    n2, m2, p2 = 5, p1, 3
    A2 = np.random.randint(-5, 5, size=(n2, n2))
    B2 = np.random.randint(-5, 5, size=(n2, m2))
    C2 = np.random.randint(-5, 5, size=(p2, n2))
    D2 = np.random.randint(-5, 5, size=(p2, m2))
    sys2 = LinearSystem(A2, B2, C2, D2)
    sys = FeedbackSystem(sys1, sys2)
    linsys = sys.linearize()
    assert np.array_equal(
        linsys.A, np.block([[A1 + B1 @ D2 @ C1, B1 @ C2], [B2 @ C1, A2]])
    )
    assert np.array_equal(linsys.B, np.block([[B1], [np.zeros((n2, m1))]]))
    assert np.array_equal(linsys.C, np.block([[C1, np.zeros((p1, n2))]]))
    assert np.array_equal(linsys.D, np.zeros((p1, m1)))

