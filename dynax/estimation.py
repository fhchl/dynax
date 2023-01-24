from dataclasses import field, fields
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.signal as sig
from jax.tree_util import tree_flatten, tree_map, tree_structure
from jaxtyping import Array, PyTree
from scipy.optimize import least_squares
from scipy.optimize._optimize import MemoizeJac

from .system import (
    DiscreteForwardModel,
    DynamicalSystem,
    ForwardModel,
    InterpolationFunction,
    spline_it,
)
from .util import value_and_jacfwd


def boxed_field(lower: float, upper: float, **kwargs):
    """Mark a dataclass field as having a box-constrained value."""
    try:
        metadata = dict(kwargs["metadata"])
    except KeyError:
        metadata = kwargs["metadata"] = {}
    if "constrained" in metadata:
        raise ValueError("Cannot use metadata if `constrained` already set.")
    metadata["constrained"] = ("boxed", (lower, upper))
    return field(**kwargs)


def non_negative_field(min_val=0.0, **kwargs):
    """Mark a dataclass field as having a non-negative value."""
    return boxed_field(lower=min_val, upper=np.inf, **kwargs)


def _append_flattend(a, b):
    if isinstance(b, (list, tuple)):
        a += b
    elif isinstance(b, float):
        a.append(b)
    else:
        raise ValueError


# TODO: In the best case, this function should return PyTrees of the same form
#       as `self`. Right now however, the function is NOT recursive, so won't
#       work on e.g. ForwardModel.
def build_bounds(self: DynamicalSystem) -> Tuple[PyTree, PyTree]:
    """Build PyTrees of lower and upper bounds."""
    lower_bounds = []
    upper_bounds = []
    for field_ in fields(self):
        name = field_.name
        try:
            value = self.__dict__[name]
        except KeyError:
            continue
        # static parameters have no bounds
        if field_.metadata.get("static", False):
            continue
        # dynamic value has bounds
        elif bound := field_.metadata.get("constrained", False):
            kind, aux = bound
            if kind == "boxed":
                lower, upper = aux
                _append_flattend(lower_bounds, tree_map(lambda _: lower, value))
                _append_flattend(upper_bounds, tree_map(lambda _: upper, value))
            else:
                raise ValueError("Unknown bound type {kind}.")
        # dynamic value is unbounded
        else:
            _append_flattend(lower_bounds, tree_map(lambda _: -np.inf, value))
            _append_flattend(upper_bounds, tree_map(lambda _: np.inf, value))
    treedef = tree_structure(self)
    return (
        treedef.unflatten(tuple(lower_bounds)),
        treedef.unflatten(tuple(upper_bounds)),
    )


def fit_ml(*args, **kwargs):
    import warnings

    warnings.warn("fit_ml is depiciated. Use fit_least_squares instead.")
    return fit_least_squares(*args, **kwargs)


def fit_least_squares(
    model: ForwardModel | DiscreteForwardModel,
    t: Array,
    y: Array,
    x0: Array,
    u: Callable[[float], Array] | Array | None = None,
    **kwargs,
) -> ForwardModel | DiscreteForwardModel:
    """Fit forward model with nonlinear least-squares."""
    t = jnp.asarray(t)
    y = jnp.asarray(y)
    if (
        isinstance(model, ForwardModel)
        and u is not None
        and not isinstance(u, InterpolationFunction)
    ):
        u = spline_it(t, u)

    # TODO: if model or any sub tree has some ndarray leaves, this does not
    # completely flatten the arrays. But a flat array is needed for `least_squares`.
    # Same problem appears for the bounds.
    init_params, treedef = tree_flatten(model)
    std_y = np.std(y, axis=0)
    bounds = tuple(map(lambda x: tree_flatten(x)[0], build_bounds(model.system)))

    def residuals(params):
        model = treedef.unflatten(params)
        _, pred_y = model(x0, t=t, u=u)
        res = ((y - pred_y) / std_y).reshape(-1)
        return res / np.sqrt(len(res))

    # compute primal and sensitivties in one forward pass
    fun = MemoizeJac(jax.jit(lambda x: value_and_jacfwd(residuals, x)))
    jac = fun.derivative
    # use instead:
    # - https://lmfit.github.io/lmfit-py/index.html
    # - https://github.com/dipolar-quantum-gases/jaxfit
    # - scipy.optimize.curve_fit
    # jaxfit and curvefit have an annoying API
    res = least_squares(
        fun, init_params, bounds=bounds, jac=jac, x_scale="jac", **kwargs
    )
    params = res.x
    return treedef.unflatten(params)


def transfer_function(sys: DynamicalSystem, **kwargs):
    """Compute transfer-function of linearized system."""
    linsys = sys.linearize(**kwargs)
    A, B, C, D = linsys.A, linsys.B, linsys.C, linsys.D

    def H(s):
        """Transfer-function at s."""
        identity = np.eye(linsys.n_states)
        phi_B = jnp.linalg.solve(s * identity - A, B)
        return C.dot(phi_B) + D

    return H


def fit_csd_matching(
    sys: DynamicalSystem, u, y, sr, nperseg=1024, reg=0, ret_Syx=False, **kwargs
):
    """Estimate parameters of linearized system by matching cross-spectral densities."""
    if u.ndim == 1:
        u = u[:, None]
    if y.ndim == 1:
        y = y[:, None]
    f, S_uu = sig.welch(u[:, None, :], fs=sr, nperseg=nperseg, axis=0)
    f, S_yu = sig.csd(u[:, None, :], y[:, :, None], fs=sr, nperseg=nperseg, axis=0)
    s = 2 * np.pi * f * 1j
    weight = np.std(S_yu, axis=0) * np.sqrt(len(f))
    x0, treedef = jax.tree_util.tree_flatten(sys)

    def residuals(params):
        sys = treedef.unflatten(params)
        H = transfer_function(sys)
        hatG_yx = jax.vmap(H)(s)
        hatS_yu = hatG_yx * S_uu
        res = (S_yu - hatS_yu) / weight
        regterm = params / np.where(np.asarray(x0) != 0, x0, 1) * reg
        return jnp.concatenate(
            (
                jnp.real(res).reshape(-1),
                jnp.imag(res).reshape(-1),
                regterm,  # high param values may lead to stiff ODEs
            )
        )

    bounds = tuple(map(lambda x: tree_flatten(x)[0], build_bounds(sys)))
    fun = MemoizeJac(jax.jit(lambda x: value_and_jacfwd(residuals, x)))
    jac = fun.derivative
    res = least_squares(fun, x0, jac=jac, x_scale="jac", bounds=bounds, **kwargs)
    fitted_sys = treedef.unflatten(res.x.tolist())
    if ret_Syx:
        H = transfer_function(fitted_sys)
        hatS_yu = jax.vmap(H)(s) * S_uu
        return fitted_sys, (f, hatS_yu, S_yu)
    return fitted_sys
