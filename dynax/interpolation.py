import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp


class InterpolationFunction(eqx.Module):
    """Interpolating cubic-spline function."""

    path: dfx.CubicInterpolation

    def __init__(self, ts, us):
        ts = jnp.asarray(ts)
        us = jnp.asarray(us)
        assert len(ts) == us.shape[0], "time and input must have same number of samples"
        coeffs = dfx.backward_hermite_coefficients(ts, us)
        self.path = dfx.CubicInterpolation(ts, coeffs)

    def __call__(self, t):
        return self.path.evaluate(t)


def spline_it(t, u):
    """Compute interpolating cubic-spline function."""
    return InterpolationFunction(t, u)
