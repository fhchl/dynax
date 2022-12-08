import diffrax as dfx
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from dynax import (ControlAffine, DynamicalSystem, ForwardModel, LinearSystem,
                   StaticStateFeedbackSystem)
from dynax.linearize import feedback_linearize, is_controllable, relative_degree
from dynax.models import Sastry9_9, NonlinearDrag

tols = dict(rtol=1e-04, atol=1e-06)

def test_relative_degree():
  sys = NonlinearDrag(0.1, 0.1, 0.1, 0.1)
  xs = np.random.normal(size=(100, 2))
  assert relative_degree(sys, xs) == 2


def test_is_controllable():
  n = 3
  A = np.diag(np.arange(n))
  B = np.ones((n, 1));
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
    vector_field = lambda self, x, u=None, t=None: -1*x + 2*u
    output = lambda self, x, u=None, t=None: 3*x + 4*u
  sys = TestSys()
  linsys = sys.linearize()
  assert np.array_equal(linsys.A, [[-1.]])
  assert np.array_equal(linsys.B, [[2.]])
  assert np.array_equal(linsys.C, [[3.]])
  assert np.array_equal(linsys.D, [[4.]])


def test_linearize_sastry9_9():
  """Linearize should return 2d-arrays. Refererence computed by hand."""
  sys = Sastry9_9()
  linsys = sys.linearize()
  assert np.array_equal(linsys.A, [[0,  0, 0],
                                   [1,  0, 0],
                                   [1, -1, 0]])
  assert np.array_equal(linsys.B, [[1], [1], [0]])
  assert np.array_equal(linsys.C, [[0, 0, 1]])
  assert np.array_equal(linsys.D, [[0.]])


def test_feedback_linearize_sastry9_9_target_linearized():
  """Feedback linearized system equals system linearized around x0."""
  sys = Sastry9_9()
  feedbacklaw, _ = feedback_linearize(sys, reference="linearized")
  target_sys = sys.linearize()
  feedback_sys = StaticStateFeedbackSystem(sys, feedbacklaw)
  t = np.linspace(0, 1)
  u = np.sin(t)
  x0 = jnp.zeros(sys.n_states)
  npt.assert_allclose(
    ForwardModel(target_sys)(x0, t, u)[1],
    ForwardModel(feedback_sys)(x0, t, u)[1],
    **tols
  )


# FIXME: this test fails as the feedback linearized system seems to diverge.
# Highly stiff or am I doing something wrong?
#@pytest.mark.xfail(reason="known parser issue")
def test_feedback_linearize_sastry9_9_target_normal_form():
  """Feedback linearized system gives same output as normal form."""
  sys = Sastry9_9()
  feedbacklaw, _ = feedback_linearize(sys, reference="normal_form")
  target_sys = LinearSystem(
    np.array([[0, 1, 0],
              [0, 0, 1],
              [0, 0, 0]]),
    np.array([[0, 0, 1]]).T,
    np.array([[1, 0, 0]]),  # z1 = h(x) = x3
    np.zeros((1,1))
  )
  feedback_sys = StaticStateFeedbackSystem(sys, feedbacklaw)

  t = np.linspace(0, 1)
  u = np.sin(100*2*np.pi*t)
  x0 = jnp.zeros(sys.n_states)
  solver = lambda: dfx.Dopri5()
  step = lambda: dfx.PIDController(rtol=1e-6, atol=1e-9)

  npt.assert_allclose(
    ForwardModel(target_sys, solver=solver(), step=step())(x0, t, u)[1],
    ForwardModel(feedback_sys, solver=solver(), step=step())(x0, t, u)[1],
    **tols
  )


def test_feedback_linearize():
  class TestSys(ControlAffine):
    n_inputs = 1
    n_states = 1
    f = lambda self, x, u=None, t=None: -x**3
    g = lambda self, x, u=None, t=None: 2
    h = lambda self, x, u, t=None: -x
  sys = TestSys()
  compensator, linsys = feedback_linearize(sys)