import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp
from jax import Array


class InterpolationFunction(eqx.Module):
    """Interpolating cubic-spline function."""

    path: dfx.CubicInterpolation

    def __init__(self, ts, xs):
        ts = jnp.asarray(ts)
        xs = jnp.asarray(xs)
        if len(ts) != xs.shape[0]:
            raise ValueError("time and data must have same number of samples")
        coeffs = dfx.backward_hermite_coefficients(ts, xs)
        self.path = dfx.CubicInterpolation(ts, coeffs)

    def __call__(self, t):
        return self.path.evaluate(t)


def spline_it(t: Array, x: Array) -> InterpolationFunction:
    """Create an interpolating cubic-spline function.

    Args:
        t: Times.
        u: Data points with first axis having the same length as `t`.

    """
    return InterpolationFunction(t, x)
