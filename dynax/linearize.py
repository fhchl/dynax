"""Functions related to input-output linearization of nonlinear systems."""

from collections.abc import Callable
from functools import partial
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from jax import Array

from .custom_types import Scalar
from .derivative import lie_derivative
from .system import (
    _CoupledSystemMixin,
    AbstractControlAffine,
    AbstractSystem,
    DynamicStateFeedbackSystem,
    LinearSystem,
)


def relative_degree(
    sys: AbstractControlAffine, xs: Array, output: Optional[int] = None
) -> int:
    """Estimate the relative degree of a SISO control-affine system.

    Tests that the Lie derivatives of the output are zero exactly up but not including
    to the relative-degree'th order for each state in `xs`.

    Args:
        sys: Continous time control-affine system with well defined relative degree and
            single input and output.
        xs: Samples of the state space stacked along the first axis.
        output: Optional index of the output if `sys` has multiple outputs.

    Returns:
        Estimated relative degree of the system.

    """
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

    raise RuntimeError("sys has ill-defined relative degree.")


# TODO: remove?
def is_controllable(A, B) -> bool:
    """Test controllability of linear system."""
    n = A.shape[0]
    contrmat = np.hstack([np.linalg.matrix_power(A, ni).dot(B) for ni in range(n)])
    return np.linalg.matrix_rank(contrmat) == n


# TODO: Adapt to general nonlinear reference system.
def input_output_linearize(
    sys: AbstractControlAffine,
    reldeg: int,
    ref: LinearSystem,
    output: Optional[int] = None,
    asymptotic: Optional[Sequence] = None,
    reg: Optional[float] = None,
) -> Callable[[Array, Array, float], Scalar]:
    """Construct an input-output linearizing feedback law.

    Args:
        sys: Continous time control-affine system with well defined relative degree and
            single input and output.
        reldeg: Relative degree of `sys` and lower bound of relative degree of `ref`.
        ref: Linear target system with single input and output.
        output: Optional index of the output if `sys` has multiple outputs.
        asymptotic: If `None`, compute the exactly linearizing law. Otherwise, compute
            an asymptotically linearizing law. Then `asymptotic` is interpreted as the
            sequence of length `reldeg` of coefficients of the characteristic polynomial
            of the tracking error system.
        reg: Regularization parameter that controls the linearization effort. Only
            effective if asymptotic is not `None`.

    Returns:
        Feedback law `u = u(x, z, v)` that input-output linearizes the system.

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

        def feedbacklaw(x: Array, z: Array, v: float) -> Scalar:
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

        def feedbacklaw(x: Array, z: Array, v: float) -> Scalar:
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


def _propagate(f: Callable[[Array, float], Array], n: int, x: Array, u: float) -> Array:
    # Propagates system for n <= discrete_relative_degree(sys) steps."""
    def fun(x, _):
        return f(x, u), None

    xn, _ = jax.lax.scan(fun, x, jnp.arange(n))
    return xn


def discrete_relative_degree(
    sys: AbstractSystem,
    xs: Array,
    us: Array,
    output: Optional[int] = None,
):
    """Estimate the relative degree of a SISO discrete-time system.

    Tests that exactly the first relative-degree - 1 output samples are independent of
    the input for each `(x, u)` for the initial state and input samples `(xs, us)`. In
    this way, the discrete relative-degree can be interpreted as a system delay.

    Args:
        sys: Discrete-time dynamical system with well defined relative degree and
            single input and output.
        xs: Initial state samples stacked along the first axis.
        us: Initial input samples stacked along the first axis.
        output: Optional index of the output if the system has multiple outputs.

    Returns:
        The discrete-time relative degree of the system.

    See :cite:p:`leeLinearizationNonlinearControl2022{def 7.7.}`.

    """
    if sys.n_inputs not in ["scalar", 1]:
        raise ValueError("System must be single input.")
    if output is None:
        # Make sure system has single output
        if sys.n_outputs not in ["scalar", 1]:
            raise ValueError(f"Output is None, but system has {sys.n_outputs} outputs.")
        h = sys.output
    else:
        h = lambda *args, **kwargs: sys.output(*args, **kwargs)[output]

    f = sys.vector_field
    y = lambda n, x, u: h(_propagate(f, n, x, u), u)
    y_depends_u = jax.grad(y, 2)

    max_reldeg = jnp.size(sys.initial_state)
    for n in range(0, max_reldeg + 1):
        res = jax.vmap(partial(y_depends_u, n))(xs, us)
        if np.all(res == 0):
            continue
        elif np.all(res != 0):
            return n
    raise RuntimeError("sys has ill defined relative degree.")


def discrete_input_output_linearize(
    sys: AbstractSystem,
    reldeg: int,
    ref: AbstractSystem,
    output: Optional[int] = None,
    solver: Optional[optx.AbstractRootFinder] = None,
) -> Callable[[Array, Array, float, float], float]:
    """Construct the input-output linearizing feedback for a discrete-time system.

    This is similar to model-predictive control with a horizon of a single time
    step and without constraints. The reference system can be nonlinear, in
    which case the feedback law implements an exact tracking controller.

    Args:
        sys: Discrete-time dynamical system with well defined relative degree and
            single input and output.
        reldeg: Relative degree of `sys` and lower bound of relative degree of `ref`.
        ref: Discrete-time reference system.
        output: Optional index of the output if the `sys` has multiple outputs.
        solver: Root finding algorithm to solve the feedback law. Defaults to
            :py:class:`optimistix.Newton` with absolute and relative tolerance `1e-6`.

    Returns:
        Feedback law :math:`u_n = u(x_n, z_n, v_n, u_{n-1})` that input-output
        linearizes the system.

    See :cite:p:`leeLinearizationNonlinearControl2022{def 7.4.}`.

    """
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
            _output(ref.output(_propagate(ref.vector_field, reldeg, z, v)))

    def feedbacklaw(x: Array, z: Array, v: float, u_prev: float) -> float:
        def fn(u, _):
            return (
                _output(h(_propagate(f, reldeg, x, u))) - y_reldeg_ref(z, v)
            ).squeeze()

        u = optx.root_find(fn, solver, u_prev).value
        return u

    return feedbacklaw


class DiscreteLinearizingSystem(AbstractSystem, _CoupledSystemMixin):
    r"""Coupled discrete-time system of dynamics, reference and linearizing feedback.

    .. math::

        x_{n+1} &= f^{sys}(x_n, v_n)   \\
        z_{n+1} &= f^{ref}(z_n, u_n)   \\
        y_n &= v_n = v(x_n, z_n, u_n)

    where :math:`v` is such that :math:`y_n^{sys} = h^{sys}(x_n, u_n)` equals
    :math:`y^{ref}_n = h^{ref}(z_n, u_n)`.

    Args:
        sys: Discrete-time dynamical system with well defined relative degree and
            single input and output.
        ref: Discrete-time reference system.
        reldeg: Discrete relative degree of `sys` and lower bound of discrete relative
            degree of `ref`.
        fb_kwargs: Additional keyword arguments passed to
            :py:func:`discrete_input_output_linearize`.

    """

    _v: Callable

    n_inputs = "scalar"

    def __init__(
        self,
        sys: AbstractSystem,
        ref: AbstractSystem,
        reldeg: int,
        **fb_kwargs,
    ):
        if sys.n_inputs != "scalar":
            raise ValueError("Only single input systems supported.")
        self._sys1 = sys
        self._sys2 = ref
        self.initial_state = jnp.append(
            self._pack_states(self._sys1.initial_state, self._sys2.initial_state), 0.0
        )
        self._v = discrete_input_output_linearize(sys, reldeg, ref, **fb_kwargs)

    def vector_field(self, x, u=None, t=None):
        (x, z), v_last = self._unpack_states(x[:-1]), x[-1]
        v = self._v(x, z, u, v_last)
        xn = self._sys1.vector_field(x, v)
        zn = self._sys2.vector_field(z, u)
        return jnp.append(self._pack_states(xn, zn), v)

    def output(self, x, u=None, t=None):
        (x, z), v_last = self._unpack_states(x[:-1]), x[-1]
        v = self._v(x, z, u, v_last)  # NOTE: feedback law is computed twice
        return v


class LinearizingSystem(DynamicStateFeedbackSystem):
    r"""Coupled ODE of nonlinear dynamics, linear reference and linearizing feedback.

    .. math::

        ẋ &= f(x) + g(x)v   \\
        ż &= Az + Bu        \\
        y &= v = v(x, z, u)

    where :math:`v` is such that :math:`y^{sys} = h(x) + i(x)v` equals
    :math:`y^{ref} = Cz + Du`.

    Args:
        sys: Continous time control-affine system with well defined relative degree and
            single input and output.
        ref: Linear target system with single input and output.
        reldeg: Relative degree of `sys` and lower bound of relative degree of `ref`.
        fb_kwargs: Additional keyword arguments passed to
            :py:func:`input_output_linearize`.

    """

    n_inputs = "scalar"

    def __init__(
        self,
        sys: AbstractControlAffine,
        ref: LinearSystem,
        reldeg: int,
        **fb_kwargs,
    ):
        v = input_output_linearize(sys, reldeg, ref, **fb_kwargs)
        super().__init__(sys, ref, v)

    def output(self, x, u, t=None):
        x, z = self._unpack_states(x)
        v = self._v(x, z, u)
        return v
