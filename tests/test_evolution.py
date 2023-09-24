import diffrax as dfx
import jax.numpy as jnp
from scipy.signal import dlsim, dlti

from dynax import DynamicalSystem, Flow, LinearSystem, Map


tols = dict(rtol=1e-04, atol=1e-06)


class SecondOrder(DynamicalSystem):
    """Second-order, linear system with constant coefficients."""

    b: float
    c: float

    def vector_field(self, x, u=None, t=None):
        """ddx + b dx + c x = u as first order with x1=x and x2=dx."""
        x1, x2 = x
        dx1 = x2
        dx2 = -self.b * x2 - self.c * x1
        return dx1, dx2

    def output(self, x, u=None, t=None):
        x1, _ = x
        return x1


def test_forward_model_crit_damp():
    b = 2
    c = 1  # critical damping as b**2 == 4*c
    sys = SecondOrder(b, c)

    def x(t, x0, dx0):
        """Solution to critically damped linear second-order system."""
        C2 = x0
        C1 = b / 2 * C2
        return jnp.exp(-b * t / 2) * (C1 * t + C2)

    x0 = (1, 0)  # x(t=0)=1, dx(t=0)=0
    t = jnp.linspace(0, 0.1, 10)
    model = Flow(sys, step=dfx.PIDController(rtol=1e-7, atol=1e-9))
    x_pred = model(x0, t)[1]
    x_true = x(t, *x0)
    assert jnp.allclose(x_true, x_pred)



def test_discrete_forward_model():
    b = 2
    c = 1  # critical damping as b**2 == 4*c
    t = jnp.arange(50)
    u = jnp.sin(1 / len(t) * 2 * jnp.pi * t)
    x0 = jnp.array([1.0, 0.0])
    A = jnp.array([[0, 1], [-c, -b]])
    B = jnp.array([[0], [1]])
    C = jnp.array([[1, 0]])
    D = jnp.zeros((1, 1))
    # test just input
    sys = LinearSystem(A, B, C, D)
    model = Map(sys)
    x, y = model(x0, u=u)  # ours
    scipy_sys = dlti(A, B, C, D)
    _, scipy_y, scipy_x = dlsim(scipy_sys, u, x0=x0)
    assert jnp.allclose(scipy_y[:, 0], y, **tols)
    assert jnp.allclose(scipy_x, x, **tols)
    # test input and time (results should be same)
    x, y = model(x0, u=u, t=t)
    scipy_t, scipy_y, scipy_x = dlsim(scipy_sys, u, x0=x0, t=t)
    assert jnp.allclose(scipy_y[:, 0], y, **tols)
    assert jnp.allclose(scipy_x, x, **tols)
