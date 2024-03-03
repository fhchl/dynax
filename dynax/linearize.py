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
from .system import (
    _CoupledSystemMixin,
    ControlAffine,
    DynamicalSystem,
    DynamicStateFeedbackSystem,
    LinearSystem,
)


# TODO: make this a method of ControlAffine
def relative_degree(sys: ControlAffine, xs: Array, output: Optional[int] = None) -> int:
    """Estimate relative degree of system on region xs."""
    if sys.n_inputs not in ["scalar", 1]:
        raise ValueError("System must be single input.")
    if output is None:
        # Make sure system has single output
        if sys.n_outputs not in ["scalar", 1]:
            raise ValueError(f"Output is None, but system has {sys.n_outputs} outputs.")
        h = sys.h
    else:
        h = lambda *args, **kwargs: sys.h(*args, **kwargs)[output]

    max_reldeg = jnp.size(sys.initial_state)
    for n in range(0, max_reldeg + 1):
        if n == 0:
            res = jax.vmap(sys.i)(xs)
        else:
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
) -> Callable[[Array, Array, float], Array]:
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
    assert sys.n_inputs == ref.n_inputs, "systems have same input dimension"
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

        def feedbacklaw(x: Array, z: Array, v: float) -> Array:
            y_reldeg_ref = cAn.dot(z) + cAnm1b * v
            y_reldeg = Lfnh(x)
            out = (y_reldeg_ref - y_reldeg) / LgLfnm1h(x)
            return out if sys.n_inputs != "scalar" else out.squeeze()

    else:
        if len(asymptotic) != reldeg:
            raise ValueError(
                f"asymptotic must be of length {reldeg=} but, {len(asymptotic)=}"
            )

        coeffs = np.concatenate(([1], asymptotic))
        if not np.all(np.real(np.roots(coeffs)) <= 0):
            raise ValueError("Polynomial must be Hurwitz")

        alphas = asymptotic

        cAis = [c.dot(np.linalg.matrix_power(A, i)) for i in range(reldeg)]
        Lfihs = [lie_derivative(sys.f, h, i) for i in range(reldeg)]

        def feedbacklaw(x: Array, z: Array, v: float) -> Array:
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
                out = error / LgLfnm1h(x)
            else:
                l = LgLfnm1h(x)
                out = error * l / (l + reg)
            return out if sys.n_inputs != "scalar" else out.squeeze()

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


class DiscreteLinearizingSystem(DynamicalSystem, _CoupledSystemMixin):
    r"""Dynamics computing linearizing feedback as output."""

    _v: Callable

    n_inputs = "scalar"

    def __init__(
        self,
        sys: DynamicalSystem,
        refsys: DynamicalSystem,
        reldeg: int,
        **fb_kwargs,
    ):
        if sys.n_inputs != "scalar":
            raise ValueError("Only single input systems supported.")
        self._sys1 = sys
        self._sys2 = refsys
        self.initial_state = jnp.append(
            self._pack_states(self._sys1.initial_state, self._sys2.initial_state), 0.0
        )
        self._v = discrete_input_output_linearize(sys, reldeg, refsys, **fb_kwargs)

    def vector_field(self, x, u=None, t=None):
        (x, z), v_last = self._unpack_states(x[:-1]), x[-1]
        v = self._v(x, z, u, v_last)
        xn = self._sys1.vector_field(x, v)
        zn = self._sys2.vector_field(z, u)
        return jnp.append(self._pack_states(xn, zn), v)

    def output(self, x, u=None, t=None):
        (x, z), v_last = self._unpack_states(x[:-1]), x[-1]
        v = self._v(x, z, u, v_last)  # FIXME: feedback law called twice
        return v


class LinearizingSystem(DynamicStateFeedbackSystem):
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

    n_inputs = "scalar"

    def __init__(
        self,
        sys: ControlAffine,
        refsys: LinearSystem,
        reldeg: int,
        **fb_kwargs,
    ):
        v = input_output_linearize(sys, reldeg, refsys, **fb_kwargs)
        super().__init__(sys, refsys, v)

    def output(self, x, u, t=None):
        x, z = self._unpack_states(x)
        v = self._v(x, z, u)
        return v
