import functools
from typing import Callable, Literal

import equinox
import jax
import jax.numpy as jnp

from .custom_types import Array, Scalar


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


def mse(target: Array, prediction: Array, axis: int = 0) -> Scalar:
    """Compute mean-squared error."""
    return jnp.mean(jnp.abs(target - prediction) ** 2, axis=axis)


def nmse(target: Array, prediction: Array, axis: int = 0) -> Scalar:
    """Compute normalized mean-squared error."""
    return mse(target, prediction, axis) / jnp.mean(jnp.abs(target) ** 2, axis=axis)


def nrmse(target: Array, prediction: Array, axis: int = 0) -> Scalar:
    """Compute normalized root mean-squared error."""
    return jnp.sqrt(nmse(target, prediction, axis))


def _monkeypatch_pretty_print():
    # FIXME: not working anymore for new equinox versions
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
