import jax.numpy as jnp

from .custom_types import Array, Scalar
from .system import (
    AbstractSystem,
    ControlAffine,
    non_negative_field,
    static_field,
)


class SpringMassDamper(AbstractSystem):
    """Forced linear spring-mass-damper system.

    .. math:: m x'' + r x' + k x = u.

    """

    m: float
    r: float
    k: float

    initial_state = jnp.zeros(2)
    n_inputs = "scalar"

    def vector_field(self, x: Array, u: Scalar, t=None) -> Array:
        """The vector field.

        .. math:: ẋ = [x_2, (u - r x_2 - k x_1) / m]^T.

        Args:
            x: State vector.
            u: Optional input vector.

        Returns:
            State derivative.

        """
        x1, x2 = x
        return jnp.array([x2, (u - self.r * x2 - self.k * x1) / self.m])


class NonlinearDrag(ControlAffine):
    """Forced spring-mass-damper system with nonlin drag.

    .. math:: m x'' +  r x' + r_2 x'|x'| + k x = u.

    """

    r: float
    """Linear drag."""
    r2: float
    """Nonlinear drag."""
    k: float
    """Stiffness."""
    m: float
    """Mass."""
    outputs: list[int] = static_field(default_factory=lambda: [0])
    """Indeces of state vectors that are outputs. Defaults to `[0]`."""

    initial_state = jnp.zeros(2)
    n_inputs = "scalar"

    def f(self, x: Array) -> Array:
        """Constant-input part of the vector field.

        .. math: ẋ = [x_2, (-r x_2 - r_2 |x_2| x_2 - k x_1) / m]^T.

        """
        x1, x2 = x
        return jnp.array(
            [x2, (-self.r * x2 - self.r2 * jnp.abs(x2) * x2 - self.k * x1) / self.m]
        )

    def g(self, x: Array) -> Array:
        """Input-proportional part of the vector field.

        .. math: ẋ = [0, 1 / m]^T.

        """
        return jnp.array([0.0, 1.0 / self.m])

    def h(self, x: Array) -> Array:
        return x[jnp.array(self.outputs)]


class Sastry9_9(ControlAffine):
    r"""Example 9.9 in :cite:t:`sastry2013nonlinear`.

    .. math::

        x_1' &= e^{x_1} u \\
        x_2' &= x_1 + x_2^2 + e^{x_1} u \\
        x_3' &= x_1 - x_2 \\
           y &= x_3 \\

    """

    initial_state = jnp.zeros(3)
    n_inputs = "scalar"

    def f(self, x: Array) -> Array:
        return jnp.array([0.0, x[0] + x[1] ** 2, x[0] - x[1]])

    def g(self, x: Array) -> Array:
        return jnp.array([jnp.exp(x[1]), jnp.exp(x[1]), 0.0])

    def h(self, x: Array) -> Scalar:
        return x[2]


class LotkaVolterra(AbstractSystem):
    r"""The notorious predator-prey model.

    .. math::

        x_1' &= α x_1 - β x_1 x_2 \\
        x_2' &= δ x_1 x_2 - γ x_2 \\
        y &= [x_1, x_2]^T

    """

    alpha: float = non_negative_field()
    beta: float = non_negative_field()
    gamma: float = non_negative_field()
    delta: float = non_negative_field()

    initial_state = jnp.ones(2) * 0.5
    n_inputs = 0

    def vector_field(self, x, u=None, t=None):
        x, y = x
        return jnp.array(
            [self.alpha * x - self.beta * x * y, self.delta * x * y - self.gamma * y]
        )
