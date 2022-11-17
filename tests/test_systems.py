import jax.numpy as jnp
import numpy as np
import diffrax as dfx
import numpy.testing as npt

from scipy.signal import dlsim, dlti
from dynax import (DynamicalSystem, FeedbackSystem, ForwardModel, LinearSystem,
                   SeriesSystem, DiscreteForwardModel)

tols = dict(rtol=1e-04, atol=1e-06)

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
  assert np.array_equal(linsys.A, np.block([[A1, np.zeros((n1, n2))],
                                            [B2.dot(C1), A2]]))
  assert np.array_equal(linsys.B, np.block([[B1], [B2.dot(D1)]]))
  assert np.array_equal(linsys.C, np.block([[D2.dot(C1), C2]]))
  assert np.array_equal(linsys.D, D2.dot(D1))


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
  assert np.array_equal(linsys.A, np.block([[A1+B1@D2@C1, B1@C2],
                                            [B2@C1,          A2]]))
  assert np.array_equal(linsys.B, np.block([[B1], [np.zeros((n2, m1))]]))
  assert np.array_equal(linsys.C, np.block([[C1, np.zeros((p1, n2))]]))
  assert np.array_equal(linsys.D, np.zeros((p1, m1)))


class SecondOrder(DynamicalSystem):
  """Second-order, linear system with constant coefficients."""
  b: float
  c: float
  def __init__(self, b, c):
    self.b = b
    self.c = c
    self.n_states = 2
    self.n_inputs = 1

  def vector_field(self, x, u=None, t=None):
    """ddx + b dx + c x = u as first order with x1=x and x2=dx."""
    x1, x2 = x
    dx1 = x2
    dx2 = -self.b * x2 - self.c * x1
    return jnp.array([dx1, dx2])

  def output(self, x, u=None, t=None):
    x1, _ = x
    return x1


def test_forward_model_crit_damp():
  b = 2; c = 1  # critical damping as b**2 == 4*c
  sys = SecondOrder(b, c)
  def x(t, x0, dx0):
    """Solution to critically damped linear second-order system."""
    C2 = x0
    C1 = b/2*C2
    return np.exp(-b*t/2)*(C1*t + C2)
  x0 = jnp.array([1, 0])  # x(t=0)=1, dx(t=0)=0
  t = np.linspace(0, 1)
  model = ForwardModel(sys, step=dfx.PIDController(rtol=1e-7, atol=1e-9))
  x_pred = model(x0, t)[1]
  x_true = x(t, *x0)
  assert np.allclose(x_true, x_pred)


def test_forward_model_lin_sys():
  b = 2; c = 1  # critical damping as b**2 == 4*c
  uconst = 1

  A = jnp.array([[0,  1],
                 [-c, -b]])
  B = jnp.array([[0], [1]])
  C = jnp.array([[1, 0]])
  D = jnp.zeros((1, 1))
  sys = LinearSystem(A, B, C, D)

  def x(t, x0, dx0, uconst):
    """Solution to critically damped linear second-order system."""
    C2 = x0 - uconst/c
    C1 = b/2*C2
    return np.exp(-b*t/2)*(C1*t + C2) + uconst/c

  x0 = jnp.array([1, 0])  # x(t=0)=1, dx(t=0)=0
  t = np.linspace(0, 1)
  u = np.ones_like(t) * uconst
  model = ForwardModel(sys, step=dfx.PIDController(rtol=1e-7, atol=1e-9))
  x_pred = model(x0, t, u)[1]
  x_true = x(t, *x0, uconst)
  assert np.allclose(x_true, x_pred)


def test_discrete_forward_model():
  b = 2; c = 1  # critical damping as b**2 == 4*c
  t = jnp.arange(50)
  u = jnp.sin(1/len(t)*2*np.pi*t)
  x0 = jnp.array([1., 0.])
  A = jnp.array([[0,  1], [-c, -b]])
  B = jnp.array([[0], [1]])
  C = jnp.array([[1, 0]])
  D = jnp.zeros((1, 1))
  # test just input
  sys = LinearSystem(A, B, C, D)
  model = DiscreteForwardModel(sys)
  x, y = model(x0, u=u) # ours
  scipy_sys = dlti(A, B, C, D)
  _, scipy_y, scipy_x = dlsim(scipy_sys, u, x0=x0)
  npt.assert_allclose(scipy_y[:, 0], y, **tols)
  npt.assert_allclose(scipy_x, x, **tols)
  # test input and time (results should be same)
  x, y = model(x0, u=u, t=t)
  scipy_t, scipy_y, scipy_x = dlsim(scipy_sys, u, x0=x0, t=t)
  npt.assert_allclose(scipy_y[:, 0], y, **tols)
  npt.assert_allclose(scipy_x, x, **tols)



if __name__ == "__main__":
  tests = [(name, obj)
           for (name, obj) in locals().items()
           if callable(obj) and name.startswith("test_")]
  for name, test in tests:
    test()
