import typing
from typing import Callable, TypeAlias

import jaxtyping


generating_docs = getattr(typing, "GENERATING_DOCUMENTATION", False)

if typing.TYPE_CHECKING:
    # In the editor.
    from jax import Array as Array
    from jax.typing import ArrayLike as ArrayLike

    Scalar: TypeAlias = Array
    ScalarLike: TypeAlias = ArrayLike
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
    from jax import Array
    from jax.typing import ArrayLike as ArrayLike

    Scalar = jaxtyping.Shaped[Array, ""]
    ScalarLike = jaxtyping.Shaped[ArrayLike, ""]


VectorFunc: TypeAlias = Callable[[Array], Array]
ScalarFunc: TypeAlias = Callable[[Array], Scalar]
# VectorField: TypeAlias = Callable[[Array, Scalar], Array]
# OutputFunc: TypeAlias = Callable[[Array, Scalar], Array]
