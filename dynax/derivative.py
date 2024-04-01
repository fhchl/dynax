"""Various functions for computing Lie derivatives."""

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.experimental.jet import jet
from jaxtyping import Scalar


VectorFunc = Callable[[Array], Array]
ScalarFunc = Callable[[Array], Scalar]
VectorField = Callable[[Array, Scalar], Array]
OutputFunc = Callable[[Array, Scalar], Array]


def lie_derivative(f: VectorFunc, h: ScalarFunc, n: int = 1) -> ScalarFunc:
    r"""Return n-th directional derivative of h along f.

    The Lie derivative is recursively defined as

    .. math::

        L_f^0 h(x) &= h(x) \\
        L_f^n h(x) &= (\nabla_x L_f^{n-1} h)(x)^T f(x)

    """
    if n < 0:
        raise ValueError(f"n must be non-negative but is {n}")
    if n == 0:
        return h
    else:
        lie_der = lie_derivative(f, h, n=n - 1)
        return lambda x: jax.jvp(
            lie_der,
            (x,),
            (f(x),),
        )[1]


def lie_derivatives_jet(f: VectorFunc, h: ScalarFunc, n: int = 1) -> VectorFunc:
    """Compute Lie derivatives using Taylor-mode differentiation.

    See :cite:p:`robenackComputationLieDerivatives2005`.

    """
    fac = jax.scipy.special.factorial(np.arange(n + 1))

    def liefun(x: Array) -> Array:
        # Taylor coefficients of x(t) = ϕₜ(x_0)
        x_primals = [x]
        x_series = [jnp.zeros_like(x) for _ in range(n)]
        for k in range(n):
            # Taylor coefficients of z(t) = f(x(t))
            z_primals, z_series = jet(f, x_primals, (x_series,))
            z = [z_primals] + z_series
            # Build xₖ from zₖ: ẋ(t) = z(t) = f(x(t))
            x_series[k] = z[k] / (k + 1)
        # Taylor coefficients of y(t) = h(x(t)) = h(ϕₜ(x_0))
        y_primals, y_series = jet(h, x_primals, (x_series,))
        Lfh = fac * jnp.array((y_primals, *y_series))
        return Lfh

    return liefun


def lie_derivative_jet(f: VectorFunc, h: ScalarFunc, n: int = 1) -> ScalarFunc:
    """Compute first element of :py:func:`lie_derivative` using :py:func:`jax.jet`."""

    def liefun(x: Array) -> Scalar:
        return lie_derivatives_jet(f, h, n)(x)[-1]

    return liefun
