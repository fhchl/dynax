import functools

import equinox
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import PyTree


def ssmatrix(data, axis=1):
    """Convert argument to a (possibly empty) 2D state space matrix.

    The axis keyword argument makes it convenient to specify that if the input
    is a vector, it is a row (axis=1) or column (axis=0) vector.

    Parameters:
        data (array, list, or string):
            Input data defining the contents of the 2D array
        axis (0 or 1):
            If input data is 1D, which axis to use for return object.  The default
            is 1, corresponding to a row matrix.

    Returns:
        arr (2D array): with shape (0, 0) if a is empty

    """
    arr = jnp.array(data, dtype=float)
    ndim = arr.ndim
    shape = arr.shape

    # Change the shape of the array into a 2D array
    if ndim > 2:
        raise ValueError("state-space matrix must be 2-dimensional")

    elif (ndim == 2 and shape == (1, 0)) or (ndim == 1 and shape == (0,)):
        # Passed an empty matrix or empty vector; change shape to (0, 0)
        shape = (0, 0)

    elif ndim == 1:
        # Passed a row or column vector
        shape = (1, shape[0]) if axis == 1 else (shape[0], 1)

    elif ndim == 0:
        # Passed a constant; turn into a matrix
        shape = (1, 1)

    #  Create the actual object used to store the result
    return arr.reshape(shape)


def value_and_jacfwd(f, x):
    """Create a function that evaluates both fun and its foward-mode jacobian.

    Only works on ndarrays, not pytrees.
    Source: https://github.com/google/jax/pull/762#issuecomment-1002267121
    """
    pushfwd = functools.partial(jax.jvp, f, (x,))
    basis = jnp.eye(x.size, dtype=x.dtype)
    y, jac = jax.vmap(pushfwd, out_axes=(None, 1))((basis,))
    return y, jac


def value_and_jacrev(f, x):
    """Create a function that evaluates both fun and its reverse-mode jacobian.

    Only works on ndarrays, not pytrees.
    Source: https://github.com/google/jax/pull/762#issuecomment-1002267121
    """
    y, pullback = jax.vjp(f, x)
    basis = jnp.eye(y.size, dtype=y.dtype)
    jac = jax.vmap(pullback)(basis)
    return y, jac


def mse(target, prediction, axis=0):
    """Compute mean-squared error."""
    return jnp.mean(jnp.abs(target - prediction) ** 2, axis=axis)


def nmse(target, prediction, axis=0):
    """Compute normalized mean-squared error."""
    return mse(target, prediction, axis) / jnp.mean(jnp.abs(target) ** 2, axis=axis)


def nrmse(target, prediction, axis=0):
    """Compute normalized root mean-squared error."""
    return jnp.sqrt(nmse(target, prediction, axis))


def _monkeypatch_pretty_print():
    from equinox._pretty_print import named_objs, bracketed, pp, dataclasses  # noqa

    def _pformat_dataclass(obj, **kwargs):
        def field_kind(field):
            if field.metadata.get("static", False):
                return "(static)"
            elif constr := field.metadata.get("constrained", False):
                return f"({constr[0]}: {constr[1]})"
            return ""

        objs = named_objs(
            [
                (
                    field.name + field_kind(field),
                    getattr(obj, field.name, "<uninitialised>"),
                )
                for field in dataclasses.fields(obj)
                if field.repr
            ],
            **kwargs,
        )
        return bracketed(
            name=pp.text(obj.__class__.__name__),
            indent=kwargs["indent"],
            objs=objs,
            lbracket="(",
            rbracket=")",
        )

    equinox._pretty_print._pformat_dataclass = _pformat_dataclass


def pretty(tree):
    return equinox.tree_pformat(tree, short_arrays=False)


# https://gist.github.com/willwhitney/dd89cac6a5b771ccff18b06b33372c75?permalink_comment_id=4634557#gistcomment-4634557
def tree_stack(trees: list[PyTree]) -> PyTree:
    """Takes a list of trees and stacks every corresponding leaf.

    For example, given two trees ((a, b), c) and ((a', b'), c'), returns
    ((stack(a, a'), stack(b, b')), stack(c, c')).

    Useful for turning a list of objects into something you can feed to a
    vmapped function.
    """
    return jtu.tree_map(lambda *v: jnp.stack(v), *trees)


def tree_unstack(tree: PyTree) -> list[PyTree]:
    """Takes a tree and turns it into a list of trees. Inverse of tree_stack.

    For example, given a tree ((a, b), c), where a, b, and c all have first
    dimension k, will make k trees
    [((a[0], b[0]), c[0]), ..., ((a[k], b[k]), c[k])]

    Useful for turning the output of a vmapped function into normal objects.
    """
    leaves, treedef = jtu.tree_flatten(tree)
    return [treedef.unflatten(leaf) for leaf in zip(*leaves, strict=True)]
