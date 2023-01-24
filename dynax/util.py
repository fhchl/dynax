import functools

import jax
import jax.numpy as jnp


def ssmatrix(data, axis=1):
    """Convert argument to a (possibly empty) 2D state space matrix.

    The axis keyword argument makes it convenient to specify that if the input
    is a vector, it is a row (axis=1) or column (axis=0) vector.

    Parameters
    ----------
    data : array, list, or string
        Input data defining the contents of the 2D array
    axis : 0 or 1
        If input data is 1D, which axis to use for return object.  The default
        is 1, corresponding to a row matrix.

    Returns
    -------
    arr : 2D array, with shape (0, 0) if a is empty

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
