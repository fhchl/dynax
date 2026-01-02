import diffrax as dfx
import jax.numpy as jnp


def solve(solver):
    x0 = jnp.array([1.0, 1.0])
    vector_field = lambda t, x, args: -0.5 * x
    term = dfx.ODETerm(vector_field)
    sol = dfx.diffeqsolve(
        term,
        t0=0.0,
        t1=1.0,
        dt0=0.01,
        y0=x0,
        solver=solver,
        stepsize_controller=dfx.PIDController(rtol=1e-3, atol=1e-6),
    )
    return sol


solve(dfx.Dopri5())  # works
solve(dfx.Kvaerno5())  # jax.errors.TracerBoolConversionError
