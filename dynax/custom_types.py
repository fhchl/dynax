import typing
from typing import Callable, TypeAlias, Union

import jaxtyping
import numpy as np


generating_docs = getattr(typing, "GENERATING_DOCUMENTATION", False)

if typing.TYPE_CHECKING:
    # In the editor.
    from jax import Array as Array
    from jax.typing import ArrayLike as ArrayLike

    Scalar: TypeAlias = Array
    ScalarLike: TypeAlias = ArrayLike
    FloatScalarLike = Union[float, Array, np.ndarray]
elif generating_docs:
    # In the docs.
    class Scalar:
        pass

    class ScalarLike:
        pass

    class Array:
        pass

    class ArrayLike:
        pass

    FloatScalarLike = float

    for cls in (Scalar, ScalarLike, Array, ArrayLike):
        cls.__module__ = "builtins"
        cls.__qualname__ = cls.__name__
else:
    # At runtime.
    from jax import Array
    from jax.typing import ArrayLike as ArrayLike

    Scalar = jaxtyping.Shaped[Array, ""]
    ScalarLike = jaxtyping.Shaped[ArrayLike, ""]
    FloatScalarLike = jaxtyping.Float[ArrayLike, ""]


VectorFunc: TypeAlias = Callable[[Array], Array]
ScalarFunc: TypeAlias = Callable[[Array], Scalar]
# VectorField: TypeAlias = Callable[[Array, Scalar], Array]
# OutputFunc: TypeAlias = Callable[[Array, Scalar], Array]
