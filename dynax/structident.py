# TODO: This is an old draft, update and test this file.
# NOTE: extended_lie_derivative is not yet implemented in derivative.py;
#       this module will fail to import until that function is added.

import jax
import jax.numpy as jnp

from .derivative import lie_derivative


def obs_ident_mat(sys, x0, u=None, t=None):
    """Generalized observability-identifiability matrix for constant input.

    Villaverde, 2017.
    """
    params, treedef = jax.tree_util.tree_flatten(sys)

    def f(x, p):
        """Vector-field for augmented state vector xp = [x, p]."""
        model = treedef.unflatten(p)
        return model.vector_field(x, u, t)

    def g(x, p):
        """Output function for augmented state vector xp = [x, p]."""
        model = treedef.unflatten(p)
        return model.output(x, t)

    params = jnp.array(params)
    O_i = jnp.vstack(
        [
            jnp.hstack(jax.jacfwd(lie_derivative(f, g, n), (0, 1))(x0, params))
            for n in range(sys.n_states + sys.n_params)
        ]
    )

    return O_i


def extended_obs_ident_mat(sys, x0, u, t=None):
    """Generalized observability-identifiability matrix for constant input.

    Villaverde, 2017.
    """
    params, treedef = jax.tree_util.tree_flatten(sys)

    def f(x, u, p):
        """Vector-field for augmented state vector xp = [x, p]."""
        model = treedef.unflatten(p)
        return model.vector_field(x, u, t)

    def g(x, p):
        """Output function for augmented state vector xp = [x, p]."""
        model = treedef.unflatten(p)
        return model.output(x, t)

    params = jnp.array(params)
    u = jnp.array(u)
    # TODO: extended_lie_derivative is not yet implemented; this function is broken.
    lies = [
        extended_lie_derivative(f, g, n)  # type: ignore[name-defined]  # noqa: F821
        for n in range(sys.n_states + sys.n_params)
    ]
    grad_of_outputs = [jnp.hstack(jax.jacfwd(l, (0, 2))(x0, u, params)) for l in lies]
    O_i = jnp.vstack(grad_of_outputs)
    return O_i
