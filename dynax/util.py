import functools
from typing import Callable, Literal

import equinox
import jax
import jax.numpy as jnp
import numpy as np

from .custom_types import Array, ArrayLike, Scalar


def value_and_jacfwd(fun: Callable, x: Array) -> tuple[Array, Callable]:
    """Create a function that evaluates both fun and its foward-mode jacobian.

    Args:
        fun: Function whose Jacobian is to be computed.
        x: Point at which function and Jacobian is evaluated.

    From this `issue <https://github.com/google/jax/pull/762#issuecomment-1002267121>`_.
    """
    pushfwd = functools.partial(jax.jvp, fun, (x,))
    basis = jnp.eye(x.size, dtype=x.dtype)
    y, jac = jax.vmap(pushfwd, out_axes=(None, 1))((basis,))
    return y, jac


def value_and_jacrev(fun: Callable, x: Array) -> tuple[Array, Callable]:
    """Create a function that evaluates both fun and its reverse-mode jacobian.

    Args:
        fun: Function whose Jacobian is to be computed.
        x: Point at which function and Jacobian is evaluated.

    From this `issue <https://github.com/google/jax/pull/762#issuecomment-1002267121>`_.
    """
    y, pullback = jax.vjp(fun, x)
    basis = jnp.eye(y.size, dtype=y.dtype)
    jac = jax.vmap(pullback)(basis)
    return y, jac


def mse(target: ArrayLike, prediction: ArrayLike, axis: int = 0) -> Scalar:
    """Compute mean-squared error."""
    return jnp.mean(jnp.abs(target - prediction) ** 2, axis=axis)


def nmse(target: ArrayLike, prediction: ArrayLike, axis: int = 0) -> Scalar:
    """Compute normalized mean-squared error."""
    return mse(target, prediction, axis) / jnp.mean(jnp.abs(target) ** 2, axis=axis)


def nrmse(target: ArrayLike, prediction: ArrayLike, axis: int = 0) -> Scalar:
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


def broadcast_right(arr, target):
    return arr.reshape(arr.shape + (1,) * (target.ndim - arr.ndim))


def dim2shape(x: int | Literal["scalar"]) -> tuple:
    return () if x == "scalar" else (x,)


def multitone(length: int, num_tones: int, first_tone: int = 0) -> Array:
    """Create a periodic bandpassed multitone signal.

    Has flat flat spectrum and crest factor <= 2.

    From: S. Boyd, "Multitone Signals with Low Crest Factor"
          IEEE Transactions on Circuits and Systems, CAS-33(10):1018-1022, October 1986
    """
    if np.log2(num_tones) % 1 != 0:
        raise ValueError("Number of tones must be power of 2")

    def rudin_sign(k: int) -> int:
        previous_bit, sign = 1, 1
        k = k - 1
        while k > 0:
            previous_bit = k % 2
            if (k := k // 2) == 0:
                break
            if previous_bit == 1 and k % 2 == 1:
                sign = -sign
        return sign

    t = np.linspace(0, 2 * np.pi, length, endpoint=False)
    u = 0
    for k in range(1, num_tones + 1):
        u += rudin_sign(k) * np.sin((k + first_tone) * t)
    u *= np.sqrt(2 / num_tones)
    return u
