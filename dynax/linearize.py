"""Functions related to feedback linearization of nonlinear systems."""

from collections.abc import Callable
from typing import Optional

import jax
import numpy as np
from jaxtyping import Array

from .derivative import lie_derivative
from .system import ControlAffine, LinearSystem


def relative_degree(sys, xs, max_reldeg=10, output: Optional[int] = None) -> int:
    """Estimate relative degree of system on region xs."""
    # TODO: when ControlAffine has y = h(x) + i(x)u, include test for n = 0,
    # i.e. i(x) == 0 for all x in xs.
    assert sys.n_inputs == 1
    if output is None:
        assert (
            sys.n_outputs == 1
        ), f"Output is None, but system has {sys.n_outputs} outputs."
        h = sys.h
    else:
        h = lambda *args, **kwargs: sys.h(*args, **kwargs)[output]

    for n in range(1, max_reldeg + 1):
        LgLfn1h = lie_derivative(sys.g, lie_derivative(sys.f, h, n - 1))
        res = jax.vmap(LgLfn1h)(xs)
        if np.all(res == 0.0):
            continue
        elif np.all(res != 0.0):
            return n
        else:
            raise RuntimeError("sys has ill-defined relative degree.")
    raise RuntimeError("Could not estimate relative degree. Increase max_reldeg.")


def is_controllable(A, B) -> bool:
    """Test controllability of linear system."""
    n = A.shape[0]
    contrmat = np.hstack([np.linalg.matrix_power(A, ni).dot(B) for ni in range(n)])
    return np.linalg.matrix_rank(contrmat) == n


def input_output_linearize(
    sys: ControlAffine, reldeg: int, ref: LinearSystem, output: Optional[int] = None
) -> Callable[[Array, Array, float], float]:
    """Construct input-output linearizing feedback law.

    Args:
        sys: nonlinear model with single input
        reldeg: relative degree of `sys` and `ref`
        ref: target model with single input
        output: specify linearizing output if systems have multiple outputs

    Note:
        Relative degree of `ref` must be same or higher than degree of `sys`.
        Only single-input-single-output systems are currently supported.

    """
    # TODO: add options for reference `normal_form` or zeros of polynomials
    assert sys.n_inputs == ref.n_inputs == 1, "systems must be single input"

    if output is None:
        assert sys.n_outputs == ref.n_outputs == 1, "systems must be single output"
        h = sys.h
        A, b, c = ref.A, ref.B, ref.C
    else:
        h = lambda x, t=None: sys.h(x, t=t)[output]
        A, b, c = ref.A, ref.B, ref.C[output]

    Lfnh = lie_derivative(sys.f, h, reldeg)
    LgLfnm1h = lie_derivative(sys.g, lie_derivative(sys.f, h, reldeg - 1))
    cAn = c.dot(np.linalg.matrix_power(A, reldeg))
    cAnm1b = c.dot(np.linalg.matrix_power(A, reldeg - 1)).dot(b)

    def feedbacklaw(x: Array, z: Array, v: float) -> float:
        return ((-Lfnh(x) + cAn.dot(z) + cAnm1b * v) / LgLfnm1h(x)).squeeze()

    return feedbacklaw
