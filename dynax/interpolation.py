import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, PyTree


class InterpolationFunction(eqx.Module):
    """Interpolating cubic-spline function."""

    path: dfx.CubicInterpolation

    def __init__(self, ts: Array, us: PyTree):
        ts_ = jnp.asarray(ts)
        coeffs = dfx.backward_hermite_coefficients(ts_, us)
        self.path = dfx.CubicInterpolation(ts_, coeffs)

    def __call__(self, t):
        return self.path.evaluate(t)


def spline_it(t, u):
    """Compute interpolating cubic-spline function."""
    return InterpolationFunction(t, u)
