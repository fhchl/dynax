"""Functions for estimating parameters of dynamical systems."""

from dataclasses import fields
from typing import Callable, Optional, TypeVar, Union

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import scipy.signal as sig
from jax import Array
from jax.flatten_util import ravel_pytree
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares, OptimizeResult
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


Evolution = TypeVar("Evolution", bound=AbstractEvolution)


def fit_least_squares(
    model: Evolution,
    t: ArrayLike,
    y: ArrayLike,
    x0: ArrayLike,
    u: Optional[ArrayLike] = None,
    batched: bool = False,
    **kwargs,
) -> OptimizeResult:
    """Fit forward model with nonlinear least-squares.

    Args:
        model: the forward model to fit
        t: times at which `y` is given
        y: target outputs of system
        x0: initial state
        u: optional system input
        kwargs: optional parameters for `scipy.optimize.least_squares`
    Returns:
        A copy of `model` with the fitted parameters.
    """
    t = jnp.asarray(t)
    y = jnp.asarray(y)
    x0 = jnp.asarray(x0)

    if batched:
        std_y = np.std(y, axis=1, keepdims=True)
        calc_coeffs = jax.vmap(dfx.backward_hermite_coefficients)
    else:
        std_y = np.std(y, axis=0, keepdims=True)
        calc_coeffs = dfx.backward_hermite_coefficients

    ucoeffs = None
    if u is not None:
        ucoeffs = calc_coeffs(t, jnp.asarray(u))

    # NOTE: least_squares wrapper also implemented at `jaxopt.ScipyLeastSquares`

    # use ravel instead of flatten as we also want to flatten all ndarrays
    init_params, unravel = ravel_pytree(model)
    bounds = _get_bounds(model)

    def residuals(params):
        model = unravel(params)
        if batched:
            model = jax.vmap(model)
        _, pred_y = model(x0, t=t, ucoeffs=ucoeffs)
        res = ((y - pred_y) / std_y).reshape(-1)
        return res / np.sqrt(len(res))

    # compute primal and sensitivties in one forward pass
    fun = MemoizeJac(jax.jit(lambda x: value_and_jacfwd(residuals, x)))
    jac = fun.derivative
    res = least_squares(
        fun, init_params, bounds=bounds, jac=jac, x_scale="jac", **kwargs
    )
    res.x = unravel(res.x)
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

    x0s, res.model = unpack(res.x, treedef, x0s_shape)
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
