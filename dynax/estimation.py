"""Functions for estimating parameters of dynamical systems.

Parameters of `model.system` can be constrained via the `*_field` functions.
"""

from __future__ import annotations  # delayed evaluation of annotations

import warnings
from dataclasses import fields
from typing import Any, Callable, cast, Literal, Optional

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import scipy.signal as sig
from jax import Array
from jax.flatten_util import ravel_pytree
from scipy.linalg import pinvh
from scipy.optimize import least_squares, OptimizeResult
from scipy.optimize._optimize import MemoizeJac

from .custom_types import ArrayLike
from .evolution import AbstractEvolution
from .system import AbstractSystem
from .util import broadcast_right, mse, nmse, nrmse, value_and_jacfwd


def _get_bounds(module: eqx.Module) -> tuple[list[float], list[float]]:
    """Build flattened arrays of lower and upper parameter bounds."""
    lower_bounds = []
    upper_bounds = []
    for field_ in fields(module):
        name = field_.name
        value = module.__dict__.get(name, None)
        if value is None:
            continue
        # elif field_.metadata.get("static", False):
        #     continue
        elif isinstance(value, eqx.Module):
            lbs, ubs = _get_bounds(value)
            lower_bounds.extend(lbs)
            upper_bounds.extend(ubs)
        elif constraint := field_.metadata.get("constrained", False):
            assert isinstance(value, jax.Array)
            _, (lb, ub) = constraint  # type: ignore
            size = np.asarray(value).size
            lower_bounds.extend([lb] * size)
            upper_bounds.extend([ub] * size)
        elif isinstance(value, jax.Array):
            size = np.asarray(value).size
            lower_bounds.extend([-np.inf] * size)
            upper_bounds.extend([np.inf] * size)
        else:
            continue
    return list(lower_bounds), list(upper_bounds)


def _key_paths(tree: Any, root: str = "tree") -> list[str]:
    """List key_paths to free fields of pytree including elements of JAX arrays."""
    f = lambda l: l.tolist() if isinstance(l, jax.Array) else l
    flattened, _ = jtu.tree_flatten_with_path(jtu.tree_map(f, tree))
    return [f"{root}{jtu.keystr(kp)}" for kp, _ in flattened]


def _compute_covariance(
    jac, cost, absolute_sigma: bool, cov_prior: Optional[np.ndarray] = None
) -> np.ndarray:
    """Compute covariance matrix from least-squares result."""
    rsize, xsize = jac.shape
    rtol = np.finfo(float).eps * max(rsize, xsize)
    hess = jac.T @ jac
    if cov_prior is not None:
        # pcov = inv(JJ^T + Σₚ⁻¹)
        hess += np.linalg.inv(cov_prior)
    pcov = cast(np.ndarray, pinvh(hess, rtol=rtol))

    warn_cov = False
    if not absolute_sigma:
        if rsize > xsize:
            s_sq = cost / (rsize - xsize)
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

    return pcov


def _least_squares(
    f: Callable[[Array], Array],
    init_params: Array,
    bounds: tuple[ArrayLike, ArrayLike],
    reg_term: Optional[Callable[[Array], Array]] = None,
    x_scale: bool = True,
    verbose_mse: bool = True,
    **kwargs: Any,
) -> OptimizeResult:
    """Least-squares with jit, autodiff, parameter scaling and regularization."""

    if reg_term is not None:
        # Add regularization term
        _f = f
        _reg_term = reg_term  # https://github.com/python/mypy/issues/7268
        f = lambda params: jnp.concatenate((_f(params), _reg_term(params)))

    if verbose_mse:
        # Scale cost to mean-squared error
        __f = f

        def f(params):
            res = __f(params)
            return res * np.sqrt(2 / res.size)

    if x_scale:
        # Scale parameters and bounds by initial values
        norm = np.where(np.asarray(init_params) != 0, np.abs(init_params), 1)
        init_params = init_params / norm
        ___f = f
        f = lambda params: ___f(params * norm)
        bounds = (np.array(bounds[0]) / norm, np.array(bounds[1]) / norm)

    fun = MemoizeJac(eqx.filter_jit(lambda x: value_and_jacfwd(f, x)))
    jac = fun.derivative
    res = least_squares(
        fun,
        init_params,
        bounds=bounds,
        jac=jac,  # type: ignore
        x_scale="jac",  # type: ignore
        **kwargs,
    )

    if x_scale:
        # Unscale parameters
        res.x = res.x * norm

    if verbose_mse:
        # Rescale to Least Squares cost
        mse_scaling = np.sqrt(2 / res.fun.size)
        res.fun = res.fun / mse_scaling
        res.jac = res.jac / mse_scaling

    if reg_term is not None:
        # Remove regularization from residuals and Jacobian and cost
        res.fun = res.fun[: -len(init_params)]
        res.jac = res.jac[: -len(init_params)]
        res.cost = np.sum(res.fun**2) / 2

    return res


def ravel_and_bounds(pytree):
    params, static = eqx.partition(pytree, lambda x: isinstance(x, jax.Array))
    params_flat, _unravel = ravel_pytree(params)
    bounds = _get_bounds(params)

    def unravel(params_flat: np.ndarray):
        params = _unravel(params_flat)
        pytree = eqx.combine(params, static)
        return pytree

    return params_flat, bounds, unravel


def fit_least_squares(
    model: AbstractEvolution,
    t: ArrayLike,
    y: ArrayLike,
    u: Optional[ArrayLike] = None,
    batched: bool = False,
    sigma: Optional[ArrayLike] = None,
    absolute_sigma: bool = False,
    reg_val: float = 0,
    reg_bias: Optional[Literal["initial"]] = None,
    verbose_mse: bool = True,
    **kwargs,
) -> OptimizeResult:
    """Fit evolution model with regularized, box-constrained nonlinear least-squares.

    For an example, see :ref:`example-fit-ode`.

    Args:
        model: A concrete evolution object.
        t: Times signal.
        y: Output signal with time dimension along the first axis.
        u: Optional input signal with time along the first axis.
        batched: Whether `t`, `y`, and `u` have an additional first axis of equal
            length holding several trajectories. The loss is then computed over all
            trajectories.
        sigma: The measurement standard deviation which is broadcasted
            against `y`. If `None`, it is assumed that the outputs have equal
            signal-to-noise ratios.
        absolute_sigma: Whether `sigma` is used in an absolute sense and the estimated
            parameter covariance reflects these absolute values. Otherwise, only
            the relative magnitudes of the sigma values matter. See also
            :func:`scipy.optimize.curve_fit`.
        reg_val: Weight of the :math:`L_2` regularization.
        reg_bias: Substractive bias term in the :math:`L_2` regularization. If
            `initial`, uses the initial parameters.
        verbose_mse: Whether the cost is scaled to the mean-squared error during logging
            with `verbose=2`.
        kwargs: Optional parameters for :py:func:`scipy.optimize.least_squares`.

    Returns:
        A Result object with the following additional attributes.

        - `result`: Fitted model.
        - `pcov`: Covariance matrix of the predicted parameters.
        - `y_pred`: Predicted outputs.
        - `key_paths`: Paths to free parameters of the model, see
          :py:func:`jax.tree_util.tree_flatten_with_path`.
        - `mse`: Mean-squared error.
        - `nmse`: Normalized mean-squared error.
        - `nrmse`: Normalized root mean-squared error.

    """
    t = jnp.asarray(t)
    y = jnp.asarray(y)

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

    init_params, bounds, unravel = ravel_and_bounds(model)

    param_bias = 0
    if reg_bias == "initial":
        param_bias = init_params

    is_regularized = np.any(reg_val != 0)
    if is_regularized:
        cov_prior = np.diag(1 / reg_val * np.ones(len(init_params)))
        reg_term = lambda params: reg_val * (params - param_bias)
    else:
        cov_prior = None
        reg_term = None

    def residual_term(params_flat):
        model = unravel(params_flat)
        if batched:
            # this can use pmap, if batch size is smaller than CPU cores
            model = jax.vmap(model)
        # FIXME: ucoeffs not supported for Map
        _, pred_y = model(t=t, ucoeffs=ucoeffs)
        res = (y - pred_y) * weight
        return res.reshape(-1)

    res = _least_squares(
        residual_term,
        init_params,
        bounds,
        reg_term=reg_term,
        verbose_mse=verbose_mse,
        **kwargs,
    )

    res.result = unravel(res.x)
    res.pcov = _compute_covariance(res.jac, res.cost, absolute_sigma, cov_prior)
    res.y_pred = y - res.fun.reshape(y.shape) / weight
    res.key_paths = _key_paths(model, root=model.__class__.__name__)
    res.mse = np.atleast_1d(mse(y, res.y_pred))
    res.nmse = np.atleast_1d(nmse(y, res.y_pred))
    res.nrmse = np.atleast_1d(nrmse(y, res.y_pred))

    return res


def _moving_window(a: Array, size: int, stride: int):
    start_idx = jnp.arange(0, len(a) - size + 1, stride)[:, None]
    inner_idx = jnp.arange(size)[None, :]
    return a[start_idx + inner_idx]


def fit_multiple_shooting(
    model: AbstractEvolution,
    t: ArrayLike,
    y: ArrayLike,
    u: Optional[ArrayLike] = None,
    num_shots: int = 1,
    continuity_penalty: float = 0.1,
    **kwargs,
) -> OptimizeResult:
    """Fit evolution model with multiple shooting.

    Multiple shooting subdivides the training problem into shooting segments and fits
    the initial states of the segments and the model parameters by minimizing the
    output error and a continuity loss of the states along the segment boundaries.

    For an example, see :ref:`example-fit-multiple-shooting`.

    Args:
        model: Concrete evolution object.
        t: Time signal.
        y: Outputs with time dimension along the first axis.
        u: Optional inputs with time along the first axis.
        num_shots: Number of shooting segments the training problem is divided into.
            If the length of the signals is not divisible by `num_shots`, the last few
            samples are ignored.
        continuity_penalty: Weights the penalty for discontinuities of the solution
            along shooting segment boundaries.
        kwargs: Optional parameters for :py:func:`scipy.optimize.least_squares`.

    Returns:
        Result object with the following additional attributes

        - `result`: The fitted model.
        - `x0s`: The initial states of the shooting segments.
        - `ts`: The times of the segments.
        - `ts0`: The times of the segments relative to the start of each segment.
        - `us`: The inputs of the segments. Only returned if `u` is not `None`.

    """
    t = jnp.asarray(t)
    y = jnp.asarray(y)

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

    n_states = len(model.system.initial_state)

    # TODO: use numpy for everything that is not jitted
    # Divide signals into segments.
    ts = _moving_window(t, num_samples_per_segment, num_samples_per_segment - 1)
    ys = _moving_window(y, num_samples_per_segment, num_samples_per_segment - 1)
    x0s = np.zeros((num_shots - 1, n_states))

    ucoeffs = None
    if u is not None:
        us = u[:num_samples]
        us = _moving_window(us, num_samples_per_segment, num_samples_per_segment - 1)
        compute_coeffs = lambda t, u: jnp.stack(dfx.backward_hermite_coefficients(t, u))
        ucoeffs = jax.vmap(compute_coeffs)(ts, us)

    # Each segment's time starts at 0.
    ts0 = ts - ts[:, :1]

    # prepare optimization
    init_params, unravel = ravel_pytree((x0s, model))
    std_y = np.std(y, axis=0)
    parameter_bounds = _get_bounds(model)
    state_bounds = (
        (num_shots - 1) * n_states * [-np.inf],
        (num_shots - 1) * n_states * [np.inf],
    )
    bounds = (
        state_bounds[0] + parameter_bounds[0],
        state_bounds[1] + parameter_bounds[1],
    )

    def residuals(params):
        x0s, model = unravel(params)
        x0s = jnp.concatenate((model.system.initial_state[None], x0s), axis=0)
        xs_pred, ys_pred = jax.vmap(model)(t=ts0, ucoeffs=ucoeffs, initial_state=x0s)
        # output residual
        res_y = ((ys - ys_pred) / std_y).reshape(-1)
        res_y = res_y / np.sqrt(len(res_y))
        # continuity residual
        std_x = jnp.std(xs_pred, axis=(0, 1))
        res_x0 = ((x0s[1:] - xs_pred[:-1, -1]) / std_x).reshape(-1)
        res_x0 = res_x0 / np.sqrt(len(res_x0))
        return jnp.concatenate((res_y, continuity_penalty * res_x0))

    res = _least_squares(residuals, init_params, bounds, x_scale=False, **kwargs)

    x0s, res.result = unravel(res.x)
    res.x0s = jnp.concatenate((res.result.system.initial_state[None], x0s), axis=0)
    res.ts = np.asarray(ts)
    res.ts0 = np.asarray(ts0)

    if u is not None:
        res.us = np.asarray(us)

    return res


def transfer_function(
    sys: AbstractSystem, to_states: bool = False, **kwargs
) -> Callable[[complex], Array]:
    """Compute transfer-function :math:`H(s)` of the linearized system.

    Args:
        sys: Concrete dynamical system.
        to_states: Whether to return the transfer-function between input and states.
            Otherwise compute it between input and output.
        kwargs: Optional arguments for :any:`AbstractSystem.linearize`.

    Returns:
        A function that computes the transfer-function at a given complex frequency.

    """
    linsys = sys.linearize(**kwargs)
    A, B, C, D = linsys.A, linsys.B, linsys.C, linsys.D

    def H(s: complex):
        """Transfer-function at s."""
        identity = np.eye(linsys.initial_state.size)
        phi_B = jnp.linalg.solve(s * identity - A, B)
        if to_states:
            return phi_B
        return C.dot(phi_B) + D

    return H


def estimate_spectra(
    u: np.ndarray, y: np.ndarray, sr: float, nperseg: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate cross- and autospectral densities using Welch's method.

    Args:
        u: Input signal.
        y: Output signal.
        sr: Sampling rate.
        nperseg: Number of samples per segment.

    Returns:
        Tuple `(f, S_yu, S_uu)` of frequencies, cross- and autospectral densities.

    """
    u_ = np.asarray(u)
    y_ = np.asarray(y)
    # Prep for correct broadcasting in sig.csd
    if u_.ndim == 1:
        u_ = u_[:, None]
    if y_.ndim == 1:
        y_ = y_[:, None]
    f, S_yu = sig.csd(u_[:, None, :], y_[:, :, None], fs=sr, nperseg=nperseg, axis=0)
    _, S_uu = sig.welch(u, fs=sr, nperseg=nperseg, axis=0)
    # Reshape back with dimensions of arguments
    S_yu = S_yu.reshape((nperseg // 2 + 1,) + y.shape[1:] + u.shape[1:])
    S_uu = S_uu.reshape((nperseg // 2 + 1,) + u.shape[1:])
    return f, S_yu, S_uu


def fit_csd_matching(
    sys: AbstractSystem,
    u: ArrayLike,
    y: ArrayLike,
    sr: float = 1.0,
    nperseg: int = 1024,
    reg: float = 0,
    x_scale: bool = True,
    verbose_mse: bool = True,
    absolute_sigma: bool = False,
    fit_dc: bool = False,
    linearize_kwargs: dict | None = None,
    **kwargs,
) -> OptimizeResult:
    """Estimate parameters of linearized system by matching cross-spectral densities.

    Args:
        sys: Concrete dynamical system.
        u: Input signal.
        y: Output signal.
        sr: Sampling rate.
        nperseg: Number of samples per segment.
        reg: Weight of the :math:`L_2` regularization.
        x_scale: Whether parameters are scaled by the initial values.
        verbose_mse: Whether the cost is scaled to the mean-squared error during logging
            with `verbose=2`.
        absolute_sigma: Whether `sigma` is used in an absolute sense and the estimated
            parameter covariance reflects these absolute values. Otherwise, only
            the relative magnitudes of the sigma values matter. See also
            :func:`scipy.optimize.curve_fit`.
        fit_dc: Whether to fit the DC term.
        linearize_kwargs: Arguments passed to
            :py:meth:`~dynax.system.AbstractSystem.linearize`.
        kwargs: Optional parameters for `scipy.optimize.least_squares`.

    Returns:
        Result object with these additional attributes.

        - `result`: Fitted model.
        - `pcov`: Estimated covariance of the parameters.
        - `key_paths`: Paths to free parameters of the model, see
          :py:func:`jax.tree_util.tree_flatten_with_path`.
        - `mse`: Mean-squared error.
        - `nmse`: Normalized mean-squared error.
        - `nrmse`: Normalized root mean-squared error.

    """
    if linearize_kwargs is None:
        linearize_kwargs = {}

    f, Syu, Suu = estimate_spectra(u, y, sr=sr, nperseg=nperseg)

    if not fit_dc:
        # remove dc term
        f = f[1:]
        Syu = Syu[1:]
        Suu = Suu[1:]

    s = 2 * np.pi * f * 1j
    weight = 1 / np.std(Syu, axis=0)
    init_params, unravel = ravel_pytree(sys)

    is_regularized = np.any(reg != 0)
    if is_regularized:
        cov_prior = np.diag(1 / reg * np.ones(len(init_params)))
        reg_term = lambda params: params * reg
    else:
        cov_prior = None
        reg_term = None

    def residuals(params):
        sys = unravel(params)
        H = transfer_function(sys, **linearize_kwargs)
        Gyu_pred = jax.vmap(H)(s)
        Syu_pred = Gyu_pred * broadcast_right(Suu, Gyu_pred)
        r = (Syu - Syu_pred) * weight
        r = jnp.concatenate((jnp.real(r), jnp.imag(r)))
        return r.reshape(-1)

    bounds = _get_bounds(sys)
    res = _least_squares(
        residuals,
        init_params,
        bounds,
        reg_term=reg_term,
        x_scale=x_scale,
        verbose_mse=verbose_mse,
        **kwargs,
    )

    Syu_pred_real, Syu_pred_imag = res.fun[: Syu.size], res.fun[Syu.size :]
    Syu_pred = Syu - (Syu_pred_real + 1j * Syu_pred_imag).reshape(Syu.shape) / weight

    res.result = unravel(res.x)
    res.pcov = _compute_covariance(
        res.jac, res.cost, absolute_sigma, cov_prior=cov_prior
    )
    res.key_paths = _key_paths(sys, root=sys.__class__.__name__)
    res.mse = np.atleast_1d(mse(Syu, Syu_pred))
    res.nmse = np.atleast_1d(nmse(Syu, Syu_pred))
    res.nrmse = np.atleast_1d(nrmse(Syu, Syu_pred))

    return res
