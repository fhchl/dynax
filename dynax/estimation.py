"""Functions for estimating parameters of dynamical systems."""

import warnings
from dataclasses import fields
from typing import Any, Callable, Literal, Optional, Union

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import scipy.signal as sig
from jax import Array
from jax.flatten_util import ravel_pytree
from jax.typing import ArrayLike
from numpy.typing import NDArray
from scipy.linalg import svd
from scipy.optimize import least_squares, OptimizeResult as _OptimizeResult
from scipy.optimize._optimize import MemoizeJac

from .evolution import AbstractEvolution
from .system import DynamicalSystem
from .util import value_and_jacfwd


def _get_bounds(module: eqx.Module) -> tuple[list, list]:
    """Build flattened lists of lower and upper bounds."""
    lower_bounds = []
    upper_bounds = []
    for field_ in fields(module):
        name = field_.name
        value = module.__dict__.get(name, None)
        if value is None:
            continue
        elif field_.metadata.get("static", False):
            continue
        elif isinstance(value, eqx.Module):
            lbs, ubs = _get_bounds(value)
            lower_bounds.extend(lbs)
            upper_bounds.extend(ubs)
        elif constraint := field_.metadata.get("constrained", False):
            _, (lb, ub) = constraint
            size = np.asarray(value).size
            lower_bounds.extend([lb] * size)
            upper_bounds.extend([ub] * size)
        else:
            size = np.asarray(value).size
            lower_bounds.extend([-np.inf] * size)
            upper_bounds.extend([np.inf] * size)
    return lower_bounds, upper_bounds


def _key_paths(tree: Any, root: str = "tree") -> list[str]:
    """List key_paths to free fields of pytree including elements of JAX arrays."""
    f = lambda l: l.tolist() if isinstance(l, jax.Array) else l
    flattened, _ = jtu.tree_flatten_with_path(jtu.tree_map(f, tree))
    return [f"{root}{jtu.keystr(kp)}" for kp, _ in flattened]


class OptimizeResult(_OptimizeResult):
    """Represents the optimization result.

    Attributes
    ----------
    x : Evolution
        The solution of the optimization.
    success : bool
        Whether or not the optimizer exited successfully.
    status : int
        Termination status of the optimizer. Its value depends on the
        underlying solver. Refer to `message` for details.
    message : str
        Description of the cause of the termination.
    fun, jac, hess: ndarray
        Values of objective function, its Jacobian and its Hessian (if
        available). The Hessians may be approximations, see the documentation
        of the function in question.
    pcov: ndarray
        Estimate of the covariance matrix.
    hess_inv : object
        Inverse of the objective function's Hessian; may be an approximation.
        Not available for all solvers. The type of this attribute may be
        either np.ndarray or scipy.sparse.linalg.LinearOperator.
    key_paths: List of key_paths for x that index the corresponding entries in `pcov`,
        `jac`, `hess` and `hess_inv`.
    nfev, njev, nhev : int
        Number of evaluations of the objective functions and of its
        Jacobian and Hessian.
    nit : int
        Number of iterations performed by the optimizer.
    maxcv : float
        The maximum constraint violation.
    """


def fit_least_squares(
    model: AbstractEvolution,
    t: ArrayLike,
    y: ArrayLike,
    x0: ArrayLike,
    u: Optional[ArrayLike] = None,
    batched: bool = False,
    sigma: Optional[ArrayLike] = None,
    absolute_sigma: bool = False,
    reg_val: float = 0,
    reg_bias: Literal["initial"] | None = None,
    reg_weight: Literal["inv_initial"] | None = None,
    **kwargs,
) -> OptimizeResult:
    """Fit forward model with (regularized) nonlinear least-squares.

    Parameters can be constrained via the `*_field` functions.

    Args:
        model: Forward model holding initial parameter estimates
        t: Times at which `y` is given
        y: Target outputs of system
        x0: Initial state
        u: Pptional system input
        batched: If True, interpret `t`, `y`, `x0`, `u` as holding multiple
            experiments stacked along the first axis.
        sigma: A 1-D sequence with values of the standard deviation of the measurement
            error for each output of `model.system`. If None, `sigma` will be set to
            the rms values of each measurement in `y`, which makes the cost
            scale-invariant to magnitude differences between measurements.
        absolute_sigma: If True, `sigma` is used in an absolute sense and the estimated
            parameter covariance `pcov` reflects these absolute values. If False
            (default), only the relative magnitudes of the `sigma` values matter and
            `sigma` is scaled to match the sample variance of the residuals after the
            fit.
        reg_val: Weight of the l2 penalty term
        reg_bias: If "initial", bias the parameter estimates towards the values in
            `model`.
        reg_weight: If "inv_initial", weight each penalty term with the inverse
            of the initial values. Applies no weightings to parameters with zero initial
            value.
        kwargs: optional parameters for `scipy.optimize.least_squares`

    Returns:
        `OptimizeResult` as returned by `scipy.optimize.least_squares` with the
        following fields defined:

            x: `model` with estimated parameters
            cov: Covariance matrix of the parameter estimate
            key_paths: List of key_paths that index the corresponding entries in `cov`
                and `jac`

    """
    t = jnp.asarray(t)
    y = jnp.asarray(y)
    x0 = jnp.asarray(x0)

    if batched:
        # First axis holds experiments, second axis holds time.
        std_y = np.std(y, axis=1, keepdims=True)
        calc_coeffs = jax.vmap(dfx.backward_hermite_coefficients)
    else:
        # First axis holds time.
        std_y = np.std(y, axis=0, keepdims=True)
        calc_coeffs = dfx.backward_hermite_coefficients

    if sigma is None:
        weight = 1 / std_y
    else:
        sigma = np.asarray(sigma)
        weight = 1 / sigma

    if u is not None:
        u = jnp.asarray(u)
        ucoeffs = calc_coeffs(t, u)
    else:
        ucoeffs = None

    init_params, unravel = ravel_pytree(model)
    bounds = _get_bounds(model)

    param_bias = 0
    if reg_bias == "initial":
        param_bias = init_params

    if reg_val != 0 and reg_weight == "initial":
        one_or_init_param = np.where(np.asarray(init_params) != 0, init_params, 1)
        reg_val = reg_val / one_or_init_param

    is_regularized = np.any(reg_val != 0)
    res_size = y.size
    if is_regularized:
        res_size = y.size + init_params.size

    def residuals(params):
        model = unravel(params)
        if batched:
            model = jax.vmap(model)
        _, pred_y = model(x0, t=t, ucoeffs=ucoeffs)
        res = (y - pred_y) * weight
        res = res.reshape(-1)
        if is_regularized:
            res = jnp.concatenate((res, reg_val * (params - param_bias)))
        # Scale cost to mean squared error (mse) for interpretable verbose output.
        res = res * np.sqrt(2 / res_size)
        return res

    # Compute primal and sensitivties in one forward pass.
    fun = MemoizeJac(jax.jit(lambda x: value_and_jacfwd(residuals, x)))
    jac = fun.derivative
    res = least_squares(
        fun, init_params, bounds=bounds, jac=jac, x_scale="jac", **kwargs
    )

    # Unscale mse to Least-Squares cost.
    res.jac = res.jac * np.sqrt(res_size / 2)  # TODO: check this
    res.cost = res.cost * res_size / 2

    # Compute covariance matrix.
    # pcov = H^{-1} ~= inv(J^T J). Use regularized inverse.
    _, s, VT = svd(res.jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
    s = s[s > threshold]
    VT = VT[: s.size]
    pcov = np.dot(VT.T / s**2, VT)

    warn_cov = False
    if not absolute_sigma:
        if res_size > res.x.size:
            s_sq = res.cost / (res_size - res.x.size)
            pcov = pcov * s_sq
        else:
            warn_cov = True

    if np.isnan(pcov).any():
        warn_cov = True

    if warn_cov:
        pcov.fill(np.inf)
        warnings.warn(
            "Covariance of the parameters could not be estimated", stacklevel=2
        )

    res.x = unravel(res.x)
    res.pcov = pcov
    res.key_paths = _key_paths(model, root=model.__class__.__name__)

    return res


def _moving_window(a: Array, size: int, stride: int):
    start_idx = jnp.arange(0, len(a) - size + 1, stride)[:, None]
    inner_idx = jnp.arange(size)[None, :]
    return a[start_idx + inner_idx]


def fit_multiple_shooting(
    model: AbstractEvolution,
    t: ArrayLike,
    y: ArrayLike,
    x0: ArrayLike,
    u: Optional[Union[Callable[[float], Array], ArrayLike]] = None,
    num_shots: int = 1,
    continuity_penalty: float = 0.0,
    **kwargs,
) -> OptimizeResult:
    """Fit forward model with multiple shooting and nonlinear least-squares.

    Args:
        model: the forward model to fit
        t: times at which `y` is given
        y: target outputs of system
        x0: initial state
        u: a function that computes the input signal or an array input samples
            at times `t`
        num_shots: number of shots the training problem is divided into
        continuity_penalty: weights the penalty for discontinuities of the
            solution along shot boundaries
        kwargs: optional parameters for `scipy.optimize.least_squares`
    Returns:
        If `u` is not `None`, returns a tuple `(model, x0s, ts, us)`, where
        `model` is the model with fitten parameters and `x0s`, `ts`, `us` are
        the initial is an array of initial states, times, and inputs for each
        shot. Else, return only `(model, x0s, ts, us)`.
    """
    t = jnp.asarray(t)
    y = jnp.asarray(y)
    x0 = jnp.asarray(x0)

    if u is None:
        msg = (
            f"t, y must have same number of samples, but have shapes "
            f"{t.shape}, {y.shape}"
        )
        assert t.shape[0] == y.shape[0], msg
    else:
        u = jnp.asarray(u)
        msg = (
            f"t, y, u must have same number of samples, but have shapes "
            f"{t.shape}, {y.shape} and {u.shape}"
        )
        assert t.shape[0] == y.shape[0] == u.shape[0], msg

    # Compute number of samples per segment. Remove samples at end if total
    # number is not divisible by num_shots.
    num_samples = len(t)
    num_samples_per_segment = int(np.floor((num_samples + (num_shots - 1)) / num_shots))
    leftover_samples = num_samples - (num_samples_per_segment * num_shots)
    if leftover_samples:
        print("Warning: removing last ", leftover_samples, "samples.")
    num_samples -= leftover_samples
    t = t[:num_samples]
    y = y[:num_samples]

    # TODO: use numpy for everything that is not jitted
    # Divide signals into segments.
    ts = _moving_window(t, num_samples_per_segment, num_samples_per_segment - 1)
    ys = _moving_window(y, num_samples_per_segment, num_samples_per_segment - 1)
    x0s = np.broadcast_to(x0, (num_shots, len(x0))).copy()
    x0s = np.concatenate((x0[None], np.zeros((num_shots - 1, len(x0)))))

    ucoeffs = None
    if u is not None:
        us = u[:num_samples]
        us = _moving_window(us, num_samples_per_segment, num_samples_per_segment - 1)
        compute_coeffs = lambda t, u: jnp.stack(dfx.backward_hermite_coefficients(t, u))
        ucoeffs = jax.vmap(compute_coeffs)(ts, us)

    # Each segment's time starts at 0.
    ts0 = ts - ts[:, :1]

    def pack(x0s, model):
        # remove initial condition which is fixed not a parameter
        x0s = x0s[1:]
        x0s_shape = x0s.shape
        flat, unravel = ravel_pytree((x0s.flatten().tolist(), model))
        return flat, unravel, x0s_shape

    def unpack(flat, unravel, x0s_shape):
        x0s_list, model = unravel(flat)
        x0s = jnp.array(x0s_list).reshape(x0s_shape)
        # add initial condition
        x0s = jnp.concatenate((x0[None], x0s), axis=0)
        return x0s, model

    # prepare optimization
    init_params, treedef, x0s_shape = pack(x0s, model)
    std_y = np.std(y, axis=0)
    parameter_bounds = _get_bounds(model)
    state_bounds = (
        (num_shots - 1) * len(x0) * [-np.inf],
        (num_shots - 1) * len(x0) * [np.inf],
    )
    bounds = (
        state_bounds[0] + parameter_bounds[0],
        state_bounds[1] + parameter_bounds[1],
    )

    def residuals(params):
        x0s, model = unpack(params, treedef, x0s_shape)
        xs_pred, ys_pred = jax.vmap(model)(x0s, t=ts0, ucoeffs=ucoeffs)
        # output residual
        res_y = ((ys - ys_pred) / std_y).reshape(-1)
        res_y = res_y / np.sqrt(len(res_y))
        # continuity residual
        std_x = jnp.std(xs_pred, axis=(0, 1))
        res_x0 = ((x0s[1:] - xs_pred[:-1, -1]) / std_x).reshape(-1)
        res_x0 = res_x0 / np.sqrt(len(res_x0))
        return jnp.concatenate((res_y, continuity_penalty * res_x0))

    # compute primal and sensitivties in one forward pass
    fun = MemoizeJac(jax.jit(lambda x: value_and_jacfwd(residuals, x)))
    jac = fun.derivative
    res = least_squares(
        fun, init_params, bounds=bounds, jac=jac, x_scale="jac", **kwargs
    )

    x0s, res.x = unpack(res.x, treedef, x0s_shape)
    res.x0s = np.asarray(x0s)
    res.ts = np.asarray(ts)
    res.ts0 = np.asarray(ts0)

    if u is not None:
        res.us = np.asarray(us)

    return res


def transfer_function(sys: DynamicalSystem, to_states: bool = False, **kwargs):
    """Compute transfer-function of linearized system."""
    linsys = sys.linearize(**kwargs)
    A, B, C, D = linsys.A, linsys.B, linsys.C, linsys.D

    def H(s):
        """Transfer-function at s."""
        identity = np.eye(linsys.n_states)
        phi_B = jnp.linalg.solve(s * identity - A, B)
        if to_states:
            return phi_B
        return C.dot(phi_B) + D

    return H


def estimate_spectra(
    u: ArrayLike, y: ArrayLike, sr: int, nperseg: int
) -> tuple[NDArray, NDArray, NDArray]:
    """Estimate cross and autospectral densities."""
    u = np.asarray(u)
    y = np.asarray(y)
    if u.ndim == 1:
        u = u[:, None]
    if y.ndim == 1:
        y = y[:, None]
    f, S_yu = sig.csd(u[:, None, :], y[:, :, None], fs=sr, nperseg=nperseg, axis=0)
    _, S_uu = sig.welch(u[:, None, :], fs=sr, nperseg=nperseg, axis=0)
    return f, S_yu, S_uu


def fit_csd_matching(
    sys: DynamicalSystem,
    u: ArrayLike,
    y: ArrayLike,
    sr: int,
    nperseg: int = 1024,
    reg: float = 0,
    **kwargs,
) -> OptimizeResult:
    """Estimate parameters of linearized system by matching cross-spectral densities."""
    f, S_yu, S_uu = estimate_spectra(u, y, sr, nperseg)
    s = 2 * np.pi * f * 1j
    weight = np.std(S_yu, axis=0) * np.sqrt(len(f))
    x0, unravel = ravel_pytree(sys)

    def residuals(params):
        sys = unravel(params)
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

    bounds = _get_bounds(sys)
    fun = MemoizeJac(jax.jit(lambda x: value_and_jacfwd(residuals, x)))
    jac = fun.derivative
    res = least_squares(fun, x0, jac=jac, x_scale="jac", bounds=bounds, **kwargs)
    res.x = unravel(res.x)
    return res
