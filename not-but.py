import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from scipy.optimize import least_squares


class ForwardModel(eqx.Module):
    params: float
    solver: dfx.AbstractSolver = eqx.field(static=True)

    def init(self, params, solver):
        self.params = params
        self.solver = solver

    def vector_field(self, x):
        return self.params * x

    def __call__(self, t, x0):
        vector_field = lambda t, x, self: self.vector_field(x)
        term = dfx.ODETerm(vector_field)
        sol = dfx.diffeqsolve(
            term,
            t0=t[0],
            t1=t[-1],
            dt0=t[1],
            y0=x0,
            saveat=dfx.SaveAt(ts=t),
            solver=self.solver,
            stepsize_controller=dfx.PIDController(rtol=1e-3, atol=1e-6),
            args=self,
            adjoint=dfx.ForwardMode(),
        )
        return sol.ys


def fit(model: ForwardModel, t, y, x0):
    """Fit forward model via least squares."""
    init_params, treedef = jax.tree_util.tree_flatten(model)

    def residuals(params):
        model = treedef.unflatten(params)
        return y - model(t, x0)

    fun = jax.jit(residuals)
    jac = jax.jit(jax.jacfwd(residuals))
    res = least_squares(fun, init_params, jac=jac)
    return treedef.unflatten(res.x)


# data and true model
t = jnp.linspace(0, 1, 100)
x0 = 1
true_model = ForwardModel(0.5, solver=dfx.Dopri5())
x_true = true_model(t, x0)
# Works
model = fit(ForwardModel(0.1, solver=dfx.Dopri5()), t, x_true, x0)
print(model)
# Works too
model = fit(ForwardModel(0.1, solver=dfx.Kvaerno3()), t, x_true, x0)
print(model)
