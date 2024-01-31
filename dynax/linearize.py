"""Functions related to feedback linearization of nonlinear systems."""

from collections.abc import Callable
from functools import partial
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
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
    assert sys.n_inputs in ["scalar", 1]
    if output is None:
        # Make sure system has single output
        msg = f"Output is None, but system has {sys.n_outputs} outputs."
        assert sys.n_outputs in ["scalar", 1], msg
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
    reg: Optional[float] = None,
) -> Callable[[Array, Array, float], float]:
    """Construct input-output linearizing feedback law.

    Args:
        sys: nonlinear model with single input
        reldeg: relative degree of `sys` and lower bound of relative degree of `ref`
        ref: target model with single input
        output: specify linearizing output if systems have multiple outputs
        asymptotic: If `None`, compute the exactly linearizing law. Otherwise,
            a sequence of length `reldeg` defining the tracking behaviour.
        reg: parameter that control the linearization effort. Only effective if
            asymptotic is not None.

    Note:
        Relative degree of `ref` must be same or higher than degree of `sys`.
        Only single-input-single-output systems are currently supported.

    """
    assert sys.n_inputs == ref.n_inputs, "systems habe same input dimension"
    assert sys.n_inputs in [1, "scalar"]

    if output is None:
        assert sys.n_outputs == ref.n_outputs, "systems must have same output dimension"
        assert sys.n_outputs in [1, "scalar"]
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
            return (y_reldeg_ref - y_reldeg) / LgLfnm1h(x)

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
                    ai * (cAi.dot(z) - Lfih(x))
                    for ai, Lfih, cAi in zip(alphas, Lfihs, cAis, strict=True)
                ]
            )
            error = y_reldeg_ref - y_reldeg + jnp.sum(ae0s)
            if reg is None:
                return error / LgLfnm1h(x)
            else:
                l = LgLfnm1h(x)
                return error * l / (l + reg)

    return feedbacklaw


def propagate(f: Callable[[Array, float], Array], n: int, x: Array, u: float) -> Array:
    """Propagates system n steps."""
    # TODO: replace by lax.scan
    if n == 0:
        return x
    return propagate(f, n - 1, f(x, u), u)


def discrete_relative_degree(
    sys: DynamicalSystem,
    xs: Array,
    us: Array,
    max_reldeg=10,
    output: Optional[int] = None,
):
    """Estimate relative degree of discrete-time system on region xs.

    Source: Lee, Linearization of Nonlinear Control Systems (2022), Def. 7.7

    """
    f = sys.vector_field
    h = sys.output

    y_depends_u = jax.grad(lambda n, x, u: h(propagate(f, n, x, u)), 2)

    for n in range(1, max_reldeg + 1):
        res = jax.vmap(partial(y_depends_u, n))(xs, us)
        if np.all(res == 0):
            continue
        elif np.all(res != 0):
            return n
        else:
            raise RuntimeError("sys has ill defined relative degree.")
    raise RuntimeError("Could not estmate relative degree. Increase max_reldeg.")


def discrete_input_output_linearize(
    sys: DynamicalSystem,
    reldeg: int,
    ref: DynamicalSystem,
    output: Optional[int] = None,
    solver: Optional[optx.AbstractRootFinder] = None,
) -> Callable[[Array, Array, float, float], float]:
    """Construct the input-output linearizing feedback for a discrete-time system."""

    # Lee 2022, Chap. 7.4
    f = lambda x, u: sys.vector_field(x, u)
    h = sys.output
    if sys.n_inputs != ref.n_inputs != 1:
        raise ValueError("Systems must have single input.")
    if output is None:
        if not (sys.n_outputs == ref.n_outputs and sys.n_outputs in ["scalar", 1]):
            raise ValueError("Systems must be single output and `output` is None.")
        _output = lambda x: x
    else:
        _output = lambda x: x[output]

    if solver is None:
        solver = optx.Newton(rtol=1e-6, atol=1e-6)

    def y_reldeg_ref(z, v):
        if isinstance(ref, LinearSystem):
            # A little faster for the linear case (if this is not optimized by jit)
            A, b, c = ref.A, ref.B, ref.C
            A_reldeg = c.dot(np.linalg.matrix_power(A, reldeg))
            B_reldeg = c.dot(np.linalg.matrix_power(A, reldeg - 1)).dot(b)
            return _output(A_reldeg.dot(z) + B_reldeg.dot(v))
        else:
            _output(ref.output(propagate(ref.vector_field, reldeg, z, v)))

    def feedbacklaw(x: Array, z: Array, v: float, u_prev: float):
        def fn(u, args):
            return (
                _output(h(propagate(f, reldeg, x, u))) - y_reldeg_ref(z, v)
            ).squeeze()

        u = optx.root_find(fn, solver, u_prev).value
        return u

    return feedbacklaw


class DiscreteLinearizingSystem(DynamicalSystem):
    r"""Dynamics computing linearizing feedback as output."""

    sys: ControlAffine
    refsys: LinearSystem
    feedbacklaw: Callable

    n_inputs = "scalar"

    def __init__(self, sys, refsys, reldeg, linearizing_output=None):
        if sys.n_inputs != "scalar":
            raise ValueError("Only single input systems supported.")
        self.sys = sys
        self.refsys = refsys
        self.n_states = self.sys.n_states + self.refsys.n_states + 1
        self.feedbacklaw = discrete_input_output_linearize(
            sys, reldeg, refsys, linearizing_output
        )

    def vector_field(self, x, u, t=None):
        x, z, v_last = x[: self.sys.n_states], x[self.sys.n_states : -1], x[-1]
        v = self.feedbacklaw(x, z, u, v_last)
        xn = self.sys.vector_field(x, v)
        zn = self.refsys.vector_field(z, u)
        return jnp.concatenate((xn, zn, jnp.array([v])))

    def output(self, x, u, t=None):
        x, z, v_last = x[: self.sys.n_states], x[self.sys.n_states : -1], x[-1]
        v = self.feedbacklaw(x, z, u, v_last)  # FIXME: feedback law called twice
        return v


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
    feedbacklaw: Callable[[Array, Array, float], float]

    n_inputs = "scalar"

    def __init__(
        self,
        sys: ControlAffine,
        refsys: LinearSystem,
        reldeg: int,
        feedbacklaw: Optional[Callable] = None,
        linearizing_output: Optional[int] = None,
    ):
        self.sys = sys
        self.refsys = refsys
        self.n_states = (
            self.sys.n_states + self.refsys.n_states
        )  # FIXME: support "scalar"
        if callable(feedbacklaw):
            self.feedbacklaw = feedbacklaw
        else:
            self.feedbacklaw = input_output_linearize(
                sys, reldeg, refsys, linearizing_output
            )

    def vector_field(self, x, u=None, t=None):
        x, z = x[: self.sys.n_states], x[self.sys.n_states :]
        y = self.feedbacklaw(x, z, u)
        dx = self.sys.vector_field(x, y)
        dz = self.refsys.vector_field(z, u)
        return jnp.concatenate((dx, dz))

    def output(self, x, u, t=None):
        x, z = x[: self.sys.n_states], x[self.sys.n_states :]
        ur = self.feedbacklaw(x, z, u)
        return ur
