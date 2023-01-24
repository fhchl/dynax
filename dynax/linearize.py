from collections.abc import Callable

import jax
import numpy as np
from jaxtyping import Array

from .ad import lie_derivative
from .system import ControlAffine, LinearSystem


def relative_degree(sys, xs, max_reldeg=10):
    """Compute relative degree of sys on region xs."""
    # TODO: when ControlAffine has y = h(x) + i(x)u, include test for n = 0,
    # i.e. i(x) == 0 for all x in xs.
    assert sys.n_inputs == 1 and sys.n_outputs == 1, "only SISO supported"
    for n in range(1, max_reldeg + 1):
        LgLfn1h = lie_derivative(sys.g, lie_derivative(sys.f, sys.h, n - 1))
        res = jax.vmap(LgLfn1h)(xs)
        if np.all(res == 0.0):
            continue
        elif np.all(res != 0.0):
            return n
        else:
            raise RuntimeError("sys has ill-defined relative degree.")
    raise RuntimeError("Could not compute relative degree. Increase max_reldeg.")


def is_controllable(A, B):
    """Test controllability of linear system."""
    n = A.shape[0]
    contrmat = np.hstack([np.linalg.matrix_power(A, ni).dot(B) for ni in range(n)])
    return np.linalg.matrix_rank(contrmat) == n


def input_output_linearize(
    sys: ControlAffine, reldeg: int, ref: LinearSystem
) -> Callable[[Array, Array, float], float]:
    """Construct input-output linearizing feedback law.

    Note: relative degree of `ref` must be same or higher than degree of sys.
    """
    # TODO: add options for reference `normal_form` or zeros of polynomials
    assert sys.n_inputs == 1 and sys.n_outputs == 1, "sys must be SISO"
    assert ref.n_inputs == ref.n_outputs == 1, "ref must be SISO"

    Lfnh = lie_derivative(sys.f, sys.h, reldeg)
    LgLfnm1h = lie_derivative(sys.g, lie_derivative(sys.f, sys.h, reldeg - 1))
    A, b, c = ref.A, ref.B, ref.C
    cAn = c.dot(np.linalg.matrix_power(A, reldeg))
    cAnm1b = c.dot(np.linalg.matrix_power(A, reldeg - 1)).dot(b)

    def feedbacklaw(x: Array, z: Array, v: float) -> float:
        return ((-Lfnh(x) + cAn.dot(z) + cAnm1b * v) / LgLfnm1h(x)).squeeze()

    return feedbacklaw
