"""Functions related to feedback linearization of nonlinear systems."""

from collections.abc import Callable
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .derivative import lie_derivative
from .system import ControlAffine, DynamicalSystem, LinearSystem


# TODO: make this a method of ControlAffine
def relative_degree(
    sys: ControlAffine, xs, max_reldeg=10, output: Optional[int] = None
) -> int:
    """Estimate relative degree of system on region xs."""
    # TODO: when ControlAffine has y = h(x) + i(x)u, include test for n = 0,
    # i.e. i(x) == 0 for all x in xs.
    assert sys.n_inputs == 1
    if output is None:
        # Make sure system has single output
        msg = f"Output is None, but system has {sys.n_outputs} outputs."
        # TODO(pytree) check output shape somewhere if this is removed
        assert sys.n_outputs == 1, msg
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
    sys: ControlAffine,
    reldeg: int,
    ref: LinearSystem,
    output: Optional[int] = None,
    asymptotic: Optional[Sequence] = None,
) -> Callable[[Array, Array, float], float]:
    """Construct input-output linearizing feedback law.

    Args:
        sys: nonlinear model with single input
        reldeg: relative degree of `sys` and lower bound of relative degree of `ref`
        ref: target model with single input
        output: specify linearizing output if systems have multiple outputs
        asymptotic: If `None`, compute the exactly linearizing law. Otherwise,
            a sequence of length `reldeg` defining the tracking behaviour.

    Note:
        Relative degree of `ref` must be same or higher than degree of `sys`.
        Only single-input-single-output systems are currently supported.

    """
    # TODO(pytree): check at runtime?
    assert sys.n_inputs == ref.n_inputs == 1, "systems must be single input"

    if output is None:
        # TODO(pytree): check somehow the single input and output, possibly at runtime
        # or inside of feedbacklaw.
        assert sys.n_outputs == ref.n_outputs == 1, "systems must be single output"
        h = sys.h
        A, b, c = ref.A, ref.B, ref.C
    else:
        h = lambda x, t=None: sys.h(x)[output]
        A, b, c = ref.A, ref.B, ref.C[output]

    Lfnh = lie_derivative(sys.f, h, reldeg)
    LgLfnm1h = lie_derivative(sys.g, lie_derivative(sys.f, h, reldeg - 1))
    cAn = c.dot(np.linalg.matrix_power(A, reldeg))
    cAnm1b = c.dot(np.linalg.matrix_power(A, reldeg - 1)).dot(b)

    if asymptotic is None:

        def feedbacklaw(x: Array, z: Array, v: float) -> float:
            y_reldeg_ref = cAn.dot(z) + cAnm1b * v
            y_reldeg = Lfnh(x)
            return ((y_reldeg_ref - y_reldeg) / LgLfnm1h(x)).squeeze()

    else:
        msg = f"asymptotic must be of length {reldeg=} but, {len(asymptotic)=}"
        assert len(asymptotic) == reldeg, msg

        coeffs = np.concatenate(([1], asymptotic))
        msg = "Polynomial must be Hurwitz"
        assert np.all(np.real(np.roots(coeffs)) <= 0)
        alphas = asymptotic

        cAis = [c.dot(np.linalg.matrix_power(A, i)) for i in range(reldeg)]
        Lfihs = [lie_derivative(sys.f, h, i) for i in range(reldeg)]

        def feedbacklaw(x: Array, z: Array, v: float) -> float:
            y_reldeg_ref = cAn.dot(z) + cAnm1b * v
            y_reldeg = Lfnh(x)
            ae0s = jnp.array(
                [
                    ai * (Lfih(x) - cAi.dot(z))
                    for ai, Lfih, cAi in zip(alphas, Lfihs, cAis)
                ]
            )
            return ((y_reldeg_ref - y_reldeg - jnp.sum(ae0s)) / LgLfnm1h(x)).squeeze()

    return feedbacklaw


class LinearizingSystem(DynamicalSystem):
    r"""Coupled ODE of nonlinear dynamics, linear reference and io linearizing law.

    .. math::

        ẋ &= f(x) + g(x)y \\
        ż &= Az + Bu \\
        y &= h(x, z, u)

    Args:
        sys: nonlinear control affine system
        refsys: linear reference system
        reldeg: relative degree of sys and lower bound of relative degree of refsys

    """

    sys: ControlAffine
    refsys: LinearSystem
    feedbacklaw: Optional[Callable] = None

    def __init__(self, sys, refsys, reldeg, feedbacklaw=None, linearizing_output=None):
        if sys.n_inputs > 1:
            raise ValueError("Only single input systems supported.")
        self.sys = sys
        self.refsys = refsys
        self.n_outputs = self.n_inputs = 1
        self.n_states = self.sys.n_states + self.refsys.n_states
        self.feedbacklaw = feedbacklaw
        if feedbacklaw is None:
            self.feedbacklaw = input_output_linearize(
                sys, reldeg, refsys, linearizing_output
            )

    def vector_field(self, x, u=None, t=None):
        # TODO(pytree) this will be x, z = x
        x, z = x[: self.sys.n_states], x[self.sys.n_states :]
        if u is None:
            u = 0.0
        y = self.feedbacklaw(x, z, u)
        dx = self.sys.vector_field(x, y)
        dz = self.refsys.vector_field(z, u)
        return jnp.concatenate((dx, dz))

    def output(self, x, u=None, t=None):
        x, z = x[: self.sys.n_states], x[self.sys.n_states :]
        y = self.feedbacklaw(x, z, u)
        return y
