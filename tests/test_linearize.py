import numpy as np
import numpy.testing as npt
import jax.numpy as jnp

from dynax import *
from dynax.linearize import feedback_linearize, is_controllable


tols = dict(rtol=1e-05, atol=1e-08)


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

class Sastry9_9(ControlAffine):
  """Sastry Example 9.9"""
  n_states = 3
  n_inputs = 1
  n_params = 0
  def f(self, x, t=None):
    x1, x2, _ = x
    return jnp.array([0., x1 + x2**2, x1-x2])
  def g(self, x, t=None):
    _, x2, _ = x
    return jnp.array([jnp.exp(x2), jnp.exp(x2), 0.])
  def h(self, x, t=None):
    _, _, x3 = x
    return x3

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
  """Feedback linearized system gives same output as system linearized around x0."""
  sys = Sastry9_9()
  feedbacklaw, _ = feedback_linearize(sys, reference="linearized")
  target_sys = sys.linearize() 
  feedback_sys = StaticStateFeedbackSystem(sys, feedbacklaw)

  t = np.linspace(0, 1)
  u = np.sin(t)
  x0 = jnp.zeros(sys.n_states)
  npt.assert_allclose(
    ForwardModel(target_sys)(t, x0, u)[1],
    ForwardModel(feedback_sys)(t, x0, u)[1],
    **tols
  )

def test_feedback_linearize_sastry9_9_target_normal_form():
  """Feedback linearized system gives same output as system linearized around x0."""
  sys = Sastry9_9()
  feedbacklaw, _ = feedback_linearize(sys, reference="linearized")
  linsys = sys.linearize() 
  feedbacksys = StaticStateFeedbackSystem(sys, feedbacklaw)

  t = np.linspace(0, 1)
  u = np.sin(t)
  x0 = jnp.zeros(sys.n_states)
  npt.assert_allclose(
    ForwardModel(linsys)(t, x0, u)[1],
    ForwardModel(feedbacksys)(t, x0, u)[1],
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



if __name__ == "__main__":
  tests = [(name, obj)
           for (name, obj) in locals().items()
           if callable(obj) and name.startswith("test_")]
  for name, test in tests:
    test()
