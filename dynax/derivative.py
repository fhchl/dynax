"""Varioius functions for computing lie derivatives."""

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import scipy.special
from jax.experimental.jet import jet


@lru_cache
def lie_derivative(f, h, n=1):
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


def lie_derivatives_jet(f, h, n=1):
    """Compute Lie derivatives using Taylor-mode differentiation.

    See :cite:p:`robenackComputationLieDerivatives2005`.

    """
    fac = scipy.special.factorial(np.arange(n + 1))

    def liefun(x):
        # taylor coefficients of x(t) = ϕₜ(x_0)
        x_primals = [x]
        x_series = [jnp.zeros_like(x) for k in range(n)]
        for k in range(n):
            # taylor coefficients of z(t) = f(x(t))
            z_primals, z_series = jet(f, x_primals, (x_series,))
            z = [z_primals] + z_series
            # build xₖ from zₖ: ẋ(t) = z(t) = f(x(t))
            x_series[k] = z[k] / (k + 1)
        # taylor coefficients of y(t) = h(x(t)) = h(ϕₜ(x_0))
        y_primals, y_series = jet(h, x_primals, (x_series,))
        Lfh = fac * jnp.array((y_primals, *y_series))
        return Lfh

    return liefun


def lie_derivative_jet(f, h, n=1):
    """Compute :py:func:`dynax.ad.lie_derivative` using `jax.jet`."""
    return lambda x: lie_derivatives_jet(f, h, n)(x)[-1]


@lru_cache
def extended_lie_derivative(f, h, n=1):
    r"""Return n-th derivative of h along f.

    .. math::

        L_f^n h(x, t) = (\nabla_x L_f^{n-1} h)(x, t)^T f(x, u, t) \\
        L_f^0 h(x, t) = h(x, t)

    """
    if n == 0:
        return lambda x, _, p: h(x, p)
    elif n == 1:
        return lambda x, u, p: jax.jacfwd(h)(x, p).dot(f(x, u[0], p))
        # FIXME: Tree structure of primal and tangential must be the same
        # return lambda x, u, p: jax.jvp(h, (x, p), (f(x, u[0], p), ))[1]
    else:
        last_lie = extended_lie_derivative(f, h, n - 1)
        grad_x = jax.jacfwd(last_lie, 0)
        grad_u = jax.jacfwd(last_lie, 1)

        def fun(x, u, p):
            uterms = min(n - 2, len(u) - 1)
            return grad_x(x, u, p).dot(f(x, u[0], p)) + grad_u(x, u, p)[:, :uterms].dot(
                u[1 : uterms + 1]
            )

        return fun
