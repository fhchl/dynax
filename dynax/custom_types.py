"""Custom type aliases used throughout dynax."""

import typing
from typing import Callable, TYPE_CHECKING, TypeAlias, Union

import jaxtyping
import numpy as np


if TYPE_CHECKING:
    # In the editor.
    BoolScalarLike = Union[bool, jaxtyping.Array, np.ndarray]
    FloatScalarLike = Union[float, jaxtyping.Array, np.ndarray]
    IntScalarLike = Union[int, jaxtyping.Array, np.ndarray]
    Scalar = jaxtyping.Array
    ScalarLike = jaxtyping.ArrayLike
    Array = jaxtyping.Array
    ArrayLike = jaxtyping.ArrayLike
elif getattr(typing, "GENERATING_DOCUMENTATION", False):
    # In the docs.
    BoolScalarLike = bool
    FloatScalarLike = float
    IntScalarLike = int
    Scalar = jaxtyping.Array
    ScalarLike = jaxtyping.ArrayLike
    Array = jaxtyping.Array
    ArrayLike = jaxtyping.ArrayLike

    for cls in (Scalar, ScalarLike, Array, ArrayLike):
        cls.__module__ = "builtins"
        cls.__qualname__ = cls.__name__
else:
    # At runtime.
    BoolScalarLike = jaxtyping.Bool[jaxtyping.ArrayLike, ""]
    FloatScalarLike = jaxtyping.Float[jaxtyping.ArrayLike, ""]
    IntScalarLike = jaxtyping.Int[jaxtyping.ArrayLike, ""]
    Scalar = jaxtyping.Shaped[jaxtyping.Array, ""]
    ScalarLike = jaxtyping.Shaped[jaxtyping.ArrayLike, ""]
    Array = jaxtyping.Array
    ArrayLike = jaxtyping.ArrayLike


RealScalarLike = Union[FloatScalarLike, IntScalarLike]

VectorFunc: TypeAlias = Callable[[Array], Array]
ScalarFunc: TypeAlias = Callable[[Array], Scalar]
# VectorField: TypeAlias = Callable[[Array, Scalar], Array]
# OutputFunc: TypeAlias = Callable[[Array, Scalar], Array]
