import jax
import jax.numpy as jnp
import numpy as np
import warnings

from collections.abc import Callable

from dynax import LinearSystem, DynamicalSystem
from dynax import lie_derivative, ControlAffine


def is_controllable(A, B):
  """Test controllability of linear system."""
  n = A.shape[0]
  contrmat = np.hstack([np.linalg.matrix_power(A, ni).dot(B) for ni in range(n)])
  return np.linalg.matrix_rank(contrmat) == n


def feedback_linearize(sys: ControlAffine, x0: np.ndarray = None, reference="linearized"
                       ) -> tuple[Callable[[float, jnp.ndarray], float], LinearSystem]:
  """Contstruct linearizing feedback law."""
  # feedback_linearization should solve the system of partial differential
  # equations to find the output, under which the system is
  # feedback/full-state/state linearizable. Right now, we assume that the
  # chosen output is already useful for feedback linearization and we assume
  # that the relative degree is equal to the number of states. 

  assert sys.n_inputs == 1, 'only single input systems supported'

  if x0 is None:
    x0 = np.zeros(sys.n_states)

  # check controllalability around x0, Sastry 9.44
  linsys = sys.linearize(x0)
  A, b, n = linsys.A, linsys.B, linsys.A.shape[0]
  if not is_controllable(A, b):
    warnings.warn(f"Linearized system not controllable.")

  # TODO: check involutivity of distribution, Sastry 9.42

  Lfnh = lie_derivative(sys.f, sys.h, n)
  LgLfn1h = lie_derivative(sys.g, lie_derivative(sys.f, sys.h, n-1))

  if reference == "linearized":
    # Sastry 9.102
    c = linsys.C
    cAn = c.dot(np.linalg.matrix_power(A, n))
    cAnm1b = c.dot(np.linalg.matrix_power(A, n-1)).dot(b)
    def feedbacklaw(x, v):
      # FIXME: feedbacklaw should only use x and v, what to do about z?
      z = x
      return ((-Lfnh(x) + cAn.dot(z) + cAnm1b*v) / LgLfn1h(x)).squeeze()
  elif reference == "normal_form":
    # Sastry 9.34
    def feedbacklaw(x, v):
      return ((-Lfnh(x) + v) / LgLfn1h(x)).squeeze()
  elif reference == "zeros_of_polynomial":
    # Sastry 9.35
    pass
  else:
    raise ValueError(f"unknown option reference={reference}")
  
  return feedbacklaw, linsys
