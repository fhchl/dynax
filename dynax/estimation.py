"""Functions for estimating parameters of dynamical systems."""

import warnings
from collections.abc import Callable, Sequence
from dataclasses import fields
from functools import partial
from typing import Any, Literal, Optional, Union

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import scipy.signal as sig
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map
from jaxtyping import ArrayLike, PyTree, Array, Float
from numpy.typing import NDArray
from scipy.linalg import pinvh
from scipy.optimize import least_squares, OptimizeResult
from scipy.optimize._optimize import MemoizeJac
from equinox.internal import ω

from .evolution import AbstractEvolution
from .system import DynamicalSystem
from .util import mse, nmse, nrmse, value_and_jacfwd, tree_stack, tree_unstack
import lineax as lx
import equinox as eqx

NDArrayLike = Union[Array, np.ndarray]


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
            size = np.size(value)
            lower_bounds.extend([lb] * size)
            upper_bounds.extend([ub] * size)
        else:
            size = np.size(value)
            lower_bounds.extend([-np.inf] * size)
            upper_bounds.extend([np.inf] * size)
    return lower_bounds, upper_bounds


def _key_paths(tree: PyTree, root: str = "tree") -> list[str]:
    """List key_paths to free fields of pytree including elements of JAX arrays."""
    f = lambda l: l.tolist() if isinstance(l, jax.Array) else l
    flattened, _ = jtu.tree_flatten_with_path(jtu.tree_map(f, tree))
    return [f"{root}{jtu.keystr(kp)}" for kp, _ in flattened]


def _compute_covariance(
    jac, cost, absolute_sigma: bool, cov_prior: Optional[NDArray] = None
) -> NDArray:
    """Compute covariance matrix from least-squares result."""
    rsize, xsize = jac.shape
    rtol = np.finfo(float).eps * max(rsize, xsize)
    hess = jac.T @ jac
    if cov_prior is not None:
        # pcov = inv(JJ^T + Σₚ⁻¹)
        hess += np.linalg.inv(cov_prior)
    pcov = pinvh(hess, rtol=rtol)

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
    x0: Array,
    bounds: tuple[list, list],
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
        norm = np.where(np.asarray(x0) != 0, np.abs(x0), 1)
        x0 = x0 / norm
        ___f = f
        f = lambda params: ___f(params * norm)
        bounds = (np.array(bounds[0]) / norm, np.array(bounds[1]) / norm)

    fun = MemoizeJac(jax.jit(lambda x: value_and_jacfwd(f, x)))
    jac = fun.derivative
    res = least_squares(fun, x0, bounds=bounds, jac=jac, x_scale="jac", **kwargs)

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
        res.fun = res.fun[: -len(x0)]
        res.jac = res.jac[: -len(x0)]
        res.cost = np.sum(res.fun**2) / 2

    return res


# TODO(pytree): What to do here?
# What is the data, e.g. output in pytree form?
# x0 = (x, y)
# x = [np.array([x1, x2, x3, ...]), np.array([y1, y2, y3, ....])]


# could all the math with the outputs be simplified by flattening and stacking y?
def fit_least_squares(
    model: AbstractEvolution,
    t: NDArrayLike,
    y: PyTree | Sequence[PyTree],
    x0: PyTree | Sequence[PyTree],
    u: Optional[PyTree | Sequence[PyTree]] = None,
    batched: bool = False,
    sigma: Optional[PyTree] = None,
    absolute_sigma: bool = False,
    reg_val: float = 0.0,
    reg_bias: Optional[Literal["initial"]] = None,
    verbose_mse: bool = True,
    **kwargs,
) -> OptimizeResult:
    """Fit forward model with (regularized) nonlinear least-squares.

    Parameters can be constrained via the `*_field` functions.

    Args:
        model: Flow instance holding initial parameter estimates
        t: Times at which `y` is given
        y: Target outputs of system of shape (times_size, output_size)
        x0: Initial state
        u: System input
        batched: If True, interpret `y`, `x0`, `u` as holding multiple experiments of
            equal length.
            If all three arguments are arrays, the experiments should be stacked along
            their first axis and the model's `vector_field` should expect and return
            `jax.Array`s. If it expects `PyTree`s, `t`, `y`, `x0` and `u` should
            instead be `Sequence`s of equal length holding the data for each
            experiment.
        sigma: A 1-D sequence with values of the standard deviation of the measurement
            error for each output of `model.system`. If None, `sigma` will be set to
            the rms values of each measurement making the cost scale-invariant to
            magnitude differences between measurements.
        absolute_sigma: If True, `sigma` is used in an absolute sense and the estimated
            parameter covariance `pcov` reflects these absolute values. If False
            (default), only the relative magnitudes of the `sigma` values matter and
            `sigma` is scaled to match the sample variance of the residuals after the
            fit.
        reg_val: Weight of the l2 penalty term.
        reg_bias: If "initial", bias the parameter estimates towards the values in
            `model`.
        verbose_mse: Scale cost to mean-squared-error for easier interpretation.
        kwargs: Optional parameters for `scipy.optimize.least_squares`.

    Returns:
        `OptimizeResult` as returned by `scipy.optimize.least_squares` with the
        following additional attributes defined:

            result: `model` with estimated parameters.
            cov: Covariance matrix of the parameter estimate.
            y_pred: Model prediction at optimum.
            key_paths: List of key_paths that index the corresponding entries in `cov`,
                `jac`, and `x`.
            mse: Mean-squared-error.
            nmse: Normalized mean-squared-error.
            nrmse: Normalized root-mean-squared-error.

    """
    t_ = jnp.asarray(t)

    if batched:
        # Multiple experiments are expected as Sequences or stacked arrays.
        batched_inputs = [y, x0] if u is None else [y, x0, u]
        is_sequence = all(isinstance(o, Sequence) for o in batched_inputs)
        is_arrays = all(isinstance(o, (jax.Array, np.ndarray)) for o in batched_inputs)
        if is_sequence:
            convert = tree_stack
        elif is_arrays:
            convert = jnp.asarray
        else:
            raise TypeError("For batched inputs, y, x0 (and u) must have same type,"
                            f"not {type(y)}, {type(x0)} (and {type(u)})")
        time_dim = 1
        calc_coeffs = jax.vmap(dfx.backward_hermite_coefficients, in_axes=(None, 0))
    else:
        time_dim = 0
        convert = lambda x: x
        calc_coeffs = dfx.backward_hermite_coefficients

    y_ = convert(y)
    x0_ = convert(x0)
    u_ = convert(u) if u is not None else None


    if sigma is None:
        std_y = tree_map(partial(np.std, axis=time_dim, keepdims=True), y_)
        weight = (1 / std_y**ω).ω
    else:
        weight = (1 / sigma**ω).ω

    if u is not None:
        ucoeffs = calc_coeffs(t_, u_)
    else:
        ucoeffs = None

    _, unravel_y = ravel_pytree(y_)
    init_params, unravel_model = ravel_pytree(model)
    bounds = _get_bounds(model)

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


    def residual_term(params):
        model = unravel_model(params)
        # Wrap in lambda with positional arguments only for vmap
        predict = lambda x, t, u: model(x, t=t, ucoeffs=u)
        if batched:
            predict = jax.vmap(predict, in_axes=(0, None, 0))
        # FIXME: ucoeffs not supported for Map
        _, pred_y = predict(x0_, t_, ucoeffs)
        res = ((y_**ω - pred_y**ω) * weight**ω).ω
        return ravel_pytree(res)[0]

    res = _least_squares(
        residual_term,
        init_params,
        bounds,
        reg_term=reg_term,
        verbose_mse=verbose_mse,
        **kwargs,
    )

    res.fun = unravel_y(res.fun)
    res.result = unravel_model(res.x)
    res.pcov = _compute_covariance(res.jac, res.cost, absolute_sigma, cov_prior)
    res.y_pred = (y_**ω - res.fun**ω / weight**ω).ω
    res.key_paths = _key_paths(model, root=model.__class__.__name__)
    res.mse = mse(y_, res.y_pred)
    res.nmse = nmse(y_, res.y_pred)
    res.nrmse = nrmse(y_, res.y_pred)

    return res

from typing import TypeVar

T = TypeVar('T')

def _moving_window(a: T, size: int, stride: int) -> T:
    start_idx = jnp.arange(0, len(a) - size + 1, stride)[:, None]
    inner_idx = jnp.arange(size)[None, :]
    return a[start_idx + inner_idx]

TimeSignal = Float[ArrayLike, "times ..."]

def fit_multiple_shooting(
    model: AbstractEvolution,
    t: TimeSignal,
    y: PyTree[TimeSignal],  # noqa: F821
    x0: PyTree,
    u: Optional[PyTree[TimeSignal]] = None,
    num_shots: int = 1,
    continuity_penalty: float = 0.1,
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
    # Check that all arguments have the same time size
    if u is None:
        ins = (t, y)
    else:
        ins = (t, y, u)
    time_size = len(t)
    is_right_size = lambda a: jnp.size(a, 0) == time_size
    if not all(map(is_right_size, jtu.tree_flatten(ins)[0])):
        raise ValueError("Inputs must be of same length.")

    # Check that output is defined
    x_shape, y_shape = jax.eval_shape(model, x0, t, u)
    if y_shape is None:
        raise ValueError(
            "`model.system.output` seems to return `None`. "
            "Did you forget to define the output method?"
        )

    # Make sure length is devisable by num_shots.
    if time_size & num_shots != 0:
        raise ValueError("Can't cleanly devide")
    else:
        # TODO: zeropad and mask or remove samples otherwise.
        pass
    samples_per_segment = time_size // num_shots

    # Devide signals into segments
    window_with_single_overlap = lambda x: _moving_window(
        x, samples_per_segment, samples_per_segment - 1
    )
    ts = jtu.tree_map(window_with_single_overlap, t)
    ys = jtu.tree_map(window_with_single_overlap, y)
    if u is not None:
        us = jtu.tree_map(window_with_single_overlap, u)
        ucoeffs = jax.vmap(dfx.backward_hermite_coefficients)(ts, us)
    else:
        ucoeffs = us = None

    # x0s for all segments but the first
    zeros_like_repeated_along_first_axis = lambda a: jnp.tile(
        jnp.zeros_like(a), (num_shots - 1,) + (1,)*jnp.ndim(a)
    )
    x0s = jtu.tree_map(zeros_like_repeated_along_first_axis, x0)

    # Each segment's time starts at 0.
    ts0 = ts - ts[:, :1]

    # Residuals are weighted by standard deviation
    std_y = tree_map(partial(np.std, axis=0), y)
    std_ys = tree_map(lambda std, y: std, std_y, ys)

    # prepare optimization
    init_params, unravel = ravel_pytree((x0s, model))
    parameter_bounds = _get_bounds(model)
    state_bounds = (
        (num_shots - 1) * len(x0) * [-np.inf],
        (num_shots - 1) * len(x0) * [np.inf],
    )
    bounds = (
        state_bounds[0] + parameter_bounds[0],
        state_bounds[1] + parameter_bounds[1],
    )

    prepend = lambda x, xs: jnp.concatenate((jnp.asarray([x]), xs))

    def residuals(params: Array) -> Array:
        x0s, model = unravel(params)
        # Prepend known initial state
        x0s = jtu.tree_map(prepend, x0, x0s)
        # Make prediction
        xs_pred, ys_pred = jax.vmap(model)(x0s, t=ts0, ucoeffs=ucoeffs)
        # Output residual
        res_y = ((ys**ω - ys_pred**ω) / std_ys**ω).ω
        # Continuity residual
        std_along_shots_and_time = partial(jnp.std, axis=(0, 1), keepdims=True)
        std_x = jtu.tree_map(std_along_shots_and_time, xs_pred)
        normalized_overlap_error = lambda x0, xs, norm: (x0[1:] - xs[:-1, -1]) * norm
        res_x0 = jtu.tree_map(
            normalized_overlap_error,
            x0s,
            xs_pred,
            (continuity_penalty / std_x**ω).ω
        )
        return jnp.concatenate((ravel_pytree(res_y)[0], ravel_pytree(res_x0)[0]))

    res = _least_squares(residuals, init_params, bounds, x_scale=False, **kwargs)

    x0s, res.result = unravel(res.x)
    res.x0s = jtu.tree_map(prepend, x0, x0s)

    # TODO: cast everything to np.ndarray?
    res.ts = np.asarray(ts)
    res.ts0 = np.asarray(ts0)

    if u is not None:
        res.us = jtu.tree_map(np.asarray, us)

    return res


def transfer_function(
    sys: DynamicalSystem,
    x: Optional[PyTree] = None,
    u: Optional[PyTree] = None,
    t: Optional[float] = None,
    to_states: bool = False,
):
    """Compute transfer-function of linearized system."""
    x = sys.x0 if x is None else x
    if x is None:
        raise ValueError("Either x or sys.x0 must be specified.")

    linsys = sys.linearize(x, u, t)
    A, B, C, D = linsys.A, linsys.B, linsys.C, linsys.D

    # Convert to complex to make sure that structures for lx.linear_solve match
    A = (A**ω * 0.j).ω
    B = (B**ω * 0.j).ω
    in_structure = jax.eval_shape(lambda: x)

    def H(s):
        """Transfer-function at s."""
        s = jnp.asarray(s, dtype=complex)
        # out_struct = jax.eval_shape(lambda: jtu.tree_map(lambda x: x[:, 0], B))
        # Iop = lx.IdentityLinearOperator(out_struct, out_struct)
        # Aop = lx.PyTreeLinearOperator(A, out_struct)
        Iop = lx.IdentityLinearOperator(in_structure, in_structure)
        Aop = lx.FunctionLinearOperator(lambda x: tree_map(jnp.dot, A, x), in_structure)
        # Don't understand: out_axes = 1 would throw an error here, but -1 works
        sol = eqx.filter_vmap(
            lx.linear_solve, in_axes=(None, 1), out_axes=-1
        )(s * Iop - Aop, B)
        phi_B = sol.value
        if to_states:
            # X = (sI - A)^-1 B U
            return phi_B
        # Y = (C (sI - A)^-1 B + D) U
        return (jtu.tree_map(jnp.dot, C, phi_B)**ω + D**ω).ω

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
    x_scale: bool = True,
    verbose_mse: bool = True,
    absolute_sigma: bool = False,
    fit_dc: bool = False,
    **kwargs,
) -> OptimizeResult:
    """Estimate parameters of linearized system by matching cross-spectral densities."""
    f, Syu, Suu = estimate_spectra(u, y, sr, nperseg)

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
        H = transfer_function(sys)
        Gyx_pred = jax.vmap(H)(s)
        Syu_pred = Gyx_pred * Suu
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
