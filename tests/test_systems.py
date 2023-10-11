import diffrax as dfx
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import numpy.testing as npt
from equinox import tree_equal

from dynax import DynamicalSystem, FeedbackSystem, Flow, LinearSystem, SeriesSystem
from dynax.example_models import Sastry9_9


tols = dict(rtol=1e-04, atol=1e-06)

def tree_almost_equal(x, y):
    if jtu.tree_structure(x) != jtu.tree_structure(y):
        return False
    return all(jtu.tree_flatten(jtu.tree_map(np.allclose, x, y))[0])


def test_dynamical_system():
    b = 2
    c = 1  # critical damping as b**2 == 4*c
    uconst = 1

    A = jnp.array([[0, 1], [-c, -b]])
    B = jnp.array([[0], [1]])
    C = jnp.array([[1, 0]])
    D = jnp.zeros((1, 1))

    class Sys(DynamicalSystem):
        def vector_field(self, x, u, t):
            return A.dot(x) + B.dot(u)
        def output(self, x, u, t):
            return C.dot(x) + D.dot(u)
    sys = Sys()

    def x(t, x0, uconst):
        """Solution to critically damped linear second-order system."""
        C2 = x0 - uconst / c
        C1 = b / 2 * C2
        return jnp.exp(-b * t / 2) * (C1 * t + C2) + uconst / c

    x0 = jnp.array([0., 0.])  # x(t=0)=0, dx(t=0)=0
    t = jnp.linspace(0, 1)
    u = jnp.ones((len(t), 1)) * uconst
    model = Flow(sys, step=dfx.PIDController(rtol=1e-7, atol=1e-9))
    _, x_pred = model(x0, t, u)
    x_true = x(t, x0[0], uconst)
    npt.assert_array_almost_equal(x_true, x_pred[:, 0])

    linsys = sys.linearize(x0, u=u[0])
    npt.assert_array_equal(linsys.A, A)
    npt.assert_array_equal(linsys.B, B)
    npt.assert_array_equal(linsys.C, C)
    npt.assert_array_equal(linsys.D, D)

    class SysB(DynamicalSystem):
        def vector_field(self, x, u, t):
            return x
        def output(self, x, u, t):
            return x
    linsys = SysB().linearize(x=(1., 2.))
    assert linsys.A == linsys.C == ((1, 0), (0, 1))
    assert linsys.B is None and linsys.D is None

    n, m, p = 3, 2, 1
    A = np.random.normal(size=(n, n))
    B = np.random.normal(size=(n, m))
    C = np.random.normal(size=(p, n))
    D = np.random.normal(size=(p, m))
    sys = LinearSystem(A, B, C, D)
    linsys = sys.linearize(x=np.zeros(n), u=np.zeros(m))
    npt.assert_array_equal(A, linsys.A)
    npt.assert_array_equal(B, linsys.B)
    npt.assert_array_equal(C, linsys.C)
    npt.assert_array_equal(D, linsys.D)


    class TestSys(DynamicalSystem):
        n_states = 1
        n_inputs = 1
        vector_field = lambda self, x, u, t=None: -1 * x + 2 * u
        output = lambda self, x, u, t=None: 3 * x + 4 * u

    sys = TestSys()
    linsys = sys.linearize(x=0., u=0.)
    npt.assert_array_equal(linsys.A, -1)
    npt.assert_array_equal(linsys.B, 2)
    npt.assert_array_equal(linsys.C, 3)
    npt.assert_array_equal(linsys.D, 4)


def test_linearize_sastry9_9():
    """Linearize should return 2d-arrays. Refererence computed by hand."""
    sys = Sastry9_9()
    linsys = sys.linearize(x=(0., 0., 0.), u=0.)
    npt.assert_array_equal(linsys.A, [[0, 1, 1], [0, 0, -1], [0, 0, 0]])
    npt.assert_array_equal(linsys.B, [1, 1, 0])
    npt.assert_array_equal(linsys.C, [0, 0, 1])
    npt.assert_array_equal(linsys.D, 0.0)


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

    x0 = jnp.array([0., 0.])  # x(t=0)=0, dx(t=0)=0
    t = jnp.linspace(0, 1)
    u = jnp.ones((len(t), 1)) * uconst
    model = Flow(sys, step=dfx.PIDController(rtol=1e-7, atol=1e-9))
    _, x_pred = model(x0, t, u)
    x_true = x(t, x0[0], uconst)
    npt.assert_array_almost_equal(x_true, x_pred[:, 0])

    class A(DynamicalSystem):
        def vector_field(self, x, u, t):
            x1, x2 = x
            return x2, x1
        def output(self, x, u, t):
            return x

    x0 = (0.5, 0.5)
    linsys = A().linearize(x0)
    Flow(linsys)(x0, t=jnp.linspace(0, 0.5))



def test_series():
    n1, m1, p1 = 4, 1, 1
    np.random.seed(42)
    A1 = np.random.uniform(-5, 5, size=(n1, n1))
    B1 = np.random.uniform(-5, 5, size=(n1, m1))
    C1 = np.random.uniform(-5, 5, size=(p1, n1))
    D1 = np.random.uniform(-5, 5, size=(p1, m1))
    sys1 = LinearSystem(A1, B1, C1, D1)
    n2, m2, p2 = 5, p1, 3
    A2 = np.random.uniform(-5, 5, size=(n2, n2))
    B2 = np.random.uniform(-5, 5, size=(n2, m2))
    C2 = np.random.uniform(-5, 5, size=(p2, n2))
    D2 = np.random.uniform(-5, 5, size=(p2, m2))
    sys2 = LinearSystem(A2, B2, C2, D2)
    sys = SeriesSystem(sys1, sys2)
    linsys = sys.linearize(x=(np.zeros(n1), np.zeros(n2)), u=0)
    assert tree_equal(
        linsys.A, ((A1, np.zeros((n1, n2))), (B2.dot(C1), A2))
    )
    assert tree_equal(linsys.B, (B1, B2.dot(D1)))
    assert tree_equal(linsys.C, (D2.dot(C1), C2))
    assert tree_equal(linsys.D, D2.dot(D1))


def test_feedback():
    np.random.seed(42)
    n1, m1, p1 = 4, 3, 2
    A1 = np.random.uniform(-5, 5, size=(n1, n1))
    B1 = np.random.uniform(-5, 5, size=(n1, m1))
    C1 = np.random.uniform(-5, 5, size=(p1, n1))
    D1 = np.zeros((p1, m1))
    sys1 = LinearSystem(A1, B1, C1, D1)
    n2, m2, p2 = 5, p1, 3
    A2 = np.random.uniform(-5, 5, size=(n2, n2))
    B2 = np.random.uniform(-5, 5, size=(n2, m2))
    C2 = np.random.uniform(-5, 5, size=(p2, n2))
    D2 = np.random.uniform(-5, 5, size=(p2, m2))
    sys2 = LinearSystem(A2, B2, C2, D2)
    sys = FeedbackSystem(sys1, sys2)
    linsys = sys.linearize(x=(np.zeros(n1), np.zeros(n2)), u=np.zeros(m1))
    assert tree_almost_equal(
        linsys.A, ((A1 + B1 @ D2 @ C1, B1 @ C2), (B2 @ C1, A2))
    )
    assert tree_almost_equal(linsys.B, (B1, np.zeros((n2, m1))))
    assert tree_almost_equal(linsys.C, (C1, np.zeros((p1, n2))))
    assert tree_almost_equal(linsys.D, np.zeros((p1, m1)))

