import jax
import jax.numpy as jnp
import numpy as np
import warnings

from collections.abc import Callable

from dynax import LinearSystem
from dynax import lie_derivative, ControlAffine

def is_controllable(A, B):
  "Test controllability of linear system."
  contrmat = np.hstack([np.linalg.matrix_power(A, ni).dot(B) for ni in range(n)])
  n = A.shape[0]
  return np.linalg.matrix_rank(contrmat) == n

def feedback_linearize(sys: ControlAffine, x0: np.ndarray) ->  tuple[Callable[[float, jnp.ndarray], float], LinearSystem]:
  "Feedback linearize a single output control-affine system."
  # check controllalability around x0, Sastry 9.44
  linsys = sys.linearize(x0)
  A, b, n = linsys.A, linsys.B, linsys.A.shape[0]
  if not is_controllable(A, b):
    warnings.warn(f"Linearized system not controllable.")
    # FIXME: this raises error, even though LS system should be controllable.

  # TODO: check in volutivity of distribution, Sastry 9.42
  # Sastry 9.102
  c = linsys.C
  cAn = c.dot(np.linalg.matrix_power(A, n))
  cAnm1b = c.dot(np.linalg.matrix_power(A, n-1)).dot(b)
  Lfnh = lie_derivative(sys.f, sys.h, n)
  LgLfn1h = lie_derivative(sys.g, lie_derivative(sys.f, sys.h, n-1))

  def compensator(r, x, z):
    return ((-Lfnh(x) + cAn.dot(z) + cAnm1b*r) / LgLfn1h(x)).squeeze()

  return compensator, linsys
