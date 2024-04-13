import typing
from typing import Callable, TypeAlias

import jax.typing
import jaxtyping


generating_docs = getattr(typing, "GENERATING_DOCUMENTATION", False)

if typing.TYPE_CHECKING:
    # In the editor.
    from jax import Array as Array
    from jax.typing import ArrayLike as ArrayLike

    Scalar: TypeAlias = jax.Array
    ScalarLike: TypeAlias = jax.typing.ArrayLike
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

    for cls in (Scalar, ScalarLike, Array, ArrayLike):
        cls.__module__ = "builtins"
        cls.__qualname__ = cls.__name__
else:
    # At runtime.
    Array = jax.typing.Array
    ArrayLike = jax.typing.ArrayLike
    Scalar = jaxtyping.Shaped[jax.typing.Array, ""]
    ScalarLike = jaxtyping.Shaped[jax.typing.ArrayLike, ""]


VectorFunc: TypeAlias = Callable[[Array], Array]
ScalarFunc: TypeAlias = Callable[[Array], Scalar]
# VectorField: TypeAlias = Callable[[Array, Scalar], Array]
# OutputFunc: TypeAlias = Callable[[Array, Scalar], Array]
