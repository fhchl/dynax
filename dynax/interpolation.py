import diffrax as dfx
import equinox
import jax.numpy as jnp
from jax import Array


class InterpolationFunction(equinox.Module):
    """Interpolating cubic-spline function."""

    path: dfx.CubicInterpolation

    def __init__(self, ts: Array, xs: Array):
        ts = jnp.asarray(ts)
        xs = jnp.asarray(xs)
        if len(ts) != xs.shape[0]:
            raise ValueError("time and data must have same number of samples")
        coeffs = dfx.backward_hermite_coefficients(ts, xs)
        self.path = dfx.CubicInterpolation(ts, coeffs)

    def __call__(self, t: float) -> Array:
        return self.path.evaluate(t)


def spline_it(ts: Array, xs: Array) -> InterpolationFunction:
    """Create an interpolating cubic-spline function.

    Args:
        ts: Time sequence.
        xs: Data points with first axis having the same length as `t`.

    Returns:
        A function `f(t)` that computes the interpolated value at time `t`.

    """
    return InterpolationFunction(ts, xs)
