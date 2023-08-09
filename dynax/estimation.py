"""Functions for estimating parameters of dynamical systems."""

from dataclasses import fields
from typing import Callable, Optional, TypeVar, Union
import warnings

import diffrax as dfx
import equinox as eqx
import jax
from jax._src.custom_derivatives import Residuals
import jax.numpy as jnp
import numpy as np
import scipy.signal as sig
from jax import Array
from jax.typing import ArrayLike
from jax.flatten_util import ravel_pytree
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares, OptimizeResult
from scipy.optimize._optimize import MemoizeJac
from scipy.linalg import svd

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


Evolution = TypeVar("Evolution", bound=AbstractEvolution)


def _least_squares(
    fun,
    args=(),
    kwargs={},
    absolute_sigma=False,
    verbose_mse=True,
    **lskwargs,
):
    """least_squares for `equinox.Module`s with automatic differentiation."""
    # Use ravel instead of flatten as we also want to flatten all ndarrays.
    init_params, unravel = ravel_pytree(fun)
    bounds = _get_bounds(fun)

    def residuals(params):
        fun = unravel(params)
        r = fun(*args, **kwargs)
        # Scale cost to mean squared error (mse) for interpretable verbose output.
        if verbose_mse:
            r = r * np.sqrt(2 / r.size)
        return r.reshape(-1)

    # Compute primal and sensitivties in one forward pass.
    fun = MemoizeJac(jax.jit(lambda x: value_and_jacfwd(residuals, x)))
    jac = fun.derivative
    res = least_squares(fun, init_params, bounds=bounds, jac=jac, **lskwargs)

    ysize = res.fun.size

    # Unscale mse to Least-Squares cost.
    if verbose_mse:
        res.jac = res.jac * np.sqrt(ysize / 2)  # TODO: check this
        res.cost = res.cost * ysize / 2

    # pcov = H^{-1} ~= inv(J^T J). Do regularized inverse here.
    _, s, VT = svd(res.jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
    s = s[s > threshold]
    VT = VT[: s.size]
    pcov = np.dot(VT.T / s**2, VT)

    warn_cov = False
    if not absolute_sigma:
        if ysize > res.x.size:
            s_sq = res.cost / (ysize - res.x.size)
            pcov = pcov * s_sq
        else:
            warn_cov = True

    if np.isnan(pcov).any():
        warn_cov = True

    if warn_cov:
        pcov.fill(np.inf)
        warnings.warn("Covariance of the parameters could not be estimated")

    res.x = unravel(res.x)
    res.cov = unravel(list(map(lambda x: unravel(x), pcov)))
    res.jac = unravel(list(map(lambda x: unravel(x), res.jac)))

    return res


def fit_least_squares(
    model: Evolution,
    t: ArrayLike,
    y: ArrayLike,
    x0: ArrayLike,
    u: Optional[ArrayLike] = None,
    batched: bool = False,
    sigma: Optional[ArrayLike] = None,
    absolute_sigma: bool = False,
    **kwargs,
) -> OptimizeResult:
    """Fit forward model with nonlinear least-squares.

    Parameter bounds can be defined via the `*_field` functions.

    Args:
        model: the forward model to fit
        t: times at which `y` is given
        y: target outputs of system
        x0: initial state
        u: optional system input
        batched: If True, interpret `t`, `y`, `x0`, `u` as holding multiple
            experiments stacked along the first axis.
        sigma: A 1-D sequence with values of the standard deviation of the measurement
            error for each output of `model.system`. If None, `sigma` will be set to
            the rms values of each measurement in `y`, which makes the cost scale
            invariant to magnitude differences.
        absolute_sigma: If True, `sigma` is used in an absolute sense and the estimated
            parameter covariance `pcov` reflects these absolute values. If False
            (default), only the relative magnitudes of the `sigma` values matter and
            `sigma` is scaled to match the sample variance of the residuals after the
            fit.
        kwargs: optional parameters for `scipy.optimize.least_squares`

    Returns:
        `OptimizeResult` as returned by `scipy.optimize.least_squares` with the
        following fields defined:

            x: `model` with estimated parameters
            cov: The Covariance of the parameter estimate as a pytree of pytrees
            jac: The Jacobian as a pytree of pytrees

    """
    t = jnp.asarray(t)
    y = jnp.asarray(y)
    x0 = jnp.asarray(x0)

    if batched:
        # First axis holds experiments, second time.
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

    # Use ravel instead of flatten as we also want to flatten all ndarrays.
    init_params, unravel = ravel_pytree(model)
    bounds = _get_bounds(model)

    def residuals(params):
        model = unravel(params)
        if batched:
            model = jax.vmap(model)
        _, pred_y = model(x0, t=t, ucoeffs=ucoeffs)
        res = (y - pred_y) * weight
        # Scale cost to mean squared error (mse) for interpretable verbose output.
        res = res * np.sqrt(2 / y.size)
        return res.reshape(-1)

    # Compute primal and sensitivties in one forward pass.
    fun = MemoizeJac(jax.jit(lambda x: value_and_jacfwd(residuals, x)))
    jac = fun.derivative
    res = least_squares(
        fun, init_params, bounds=bounds, jac=jac, x_scale="jac", **kwargs
    )

    # Unscale mse to Least-Squares cost.
    res.jac = res.jac * np.sqrt(y.size / 2)  # TODO: check this
    res.cost = res.cost * y.size / 2

    # pcov = H^{-1} ~= inv(J^T J). Do regularized inverse here.
    _, s, VT = svd(res.jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
    s = s[s > threshold]
    VT = VT[: s.size]
    pcov = np.dot(VT.T / s**2, VT)

    warn_cov = False
    if not absolute_sigma:
        if y.size > res.x.size:
            s_sq = res.cost / (y.size - res.x.size)
            pcov = pcov * s_sq
        else:
            warn_cov = True

    if np.isnan(pcov).any():
        warn_cov = True

    if warn_cov:
        pcov.fill(np.inf)
        warnings.warn("Covariance of the parameters could not be estimated")

    # FIXME: I am stuck here. We need some way to transform the covariance matrix
    # into a Pytree of Pytrees.
    # If 
    #
    #    tree = ((np.array([[a, b], [c, d]]), e))
    #
    # then we would like
    # 
    #    cov[0][0, 0][1] == covariance(a, e)
    #    cov[0][0, 0][0] == np.ndarray([[cov(a, a), cov(a, b)],
    #                                   [cov(a, c), cov(a, d)]])
    #    cov[1][1] = cov(e, e)
    #
    # but! jnp.ndarrays can not hold leaves! or can they?
    # Idea: first unflatten and then ravel: unflatten doesn't care about the specific
    #         types, so we can make pytrees of pytrees
    
    res.x = unravel(res.x)
    res.cov = unravel(list(map(lambda x: unravel(x), pcov)))
    res.jac = unravel(list(map(lambda x: unravel(x), res.jac)))

    return res


def _moving_window(a: Array, size: int, stride: int):
    start_idx = jnp.arange(0, len(a) - size + 1, stride)[:, None]
    inner_idx = jnp.arange(size)[None, :]
    return a[start_idx + inner_idx]


def fit_multiple_shooting(
    model: Evolution,
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

    # FIXME: use numpy for everything that is not jitted
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
        # TODO: for dfx.NoAdjoint using diffrax<v0.3 computing jacobians through
        # vmap is very slow.
        # pmap needs exact number of devices:
        # xs_pred, ys_pred = jax.pmap(model)(x0s, t=ts0, ucoeffs=ucoeffs)
        # vmap is slow:
        xs_pred, ys_pred = jax.vmap(model)(x0s, t=ts0, ucoeffs=ucoeffs)
        # xmap needs axies names and seems complicated:
        # in_axes = [['shots', ...], ['shots', ...], ['shots', ...]]
        # out_axes = ['shots', ...]
        # m = lambda x, t, u: model(x, t=t, ucoeffs=u)
        # xs_pred, ys_pred = xmap(m, in_axes=in_axes, out_axes=[...])(x0s, ts0, ucoeffs)
        # just use serial map:
        # m = lambda x, t, u: model(x, t=t, ucoeffs=u)
        # xs_pred, ys_pred = zip(*list(map(m, x0s, ts0, ucoeffs)))
        # xs_pred = jnp.stack(xs_pred)
        # ys_pred = jnp.stack(ys_pred)
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
