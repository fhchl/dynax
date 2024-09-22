import jax.numpy as jnp
import numpy as np

from .custom_types import Array, Scalar
from .system import (
    AbstractControlAffine,
    AbstractSystem,
    boxed_field,
    field,
    non_negative_field,
    static_field,
)


# Define a general dynamical system by subclassing `AbstractSystem`.
class SpringMassDamper(AbstractSystem):
    """Forced linear spring-mass-damper system.

    .. math:: m x'' + r x' + k x = u.

    """

    # Define the system parameters as data fields.
    m: float = field()
    """Mass."""
    r: float = field()
    """Linear drag."""
    k: float = field()
    """Stiffness."""

    # The following two fields are aleady defined in `AbstractSystem`. Thus, their
    # type declarations can be left out.
    initial_state = np.zeros(2)
    n_inputs = "scalar"

    # Define the vector field of the system by implementing the `vector_field` method.
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

    # This class does not override the `AbstractSystem.output` method. The output is
    # then the full state vector by default.


# Systems that have a control affine structure can subclass `AbstractControlAffine` and
# implement the `f`, `g`, and `h` methods. Such systems can often be input-output
# linearized with the functions in `dynax.linearizate`.
class NonlinearDrag(AbstractControlAffine):
    """Forced spring-mass-damper system with nonlin drag.

    .. math:: m x'' +  r x' + r_2 x'|x'| + k x = u.

    """

    r: Array = field()
    """Linear drag."""
    r2: Array = field()
    """Nonlinear drag."""
    k: Array = field()
    """Stiffness."""
    m: Array = field()
    """Mass."""

    # We can define additional dataclass fields that do not represent trainable
    # model parameters using the `static_field` function. This function tells JAX that
    # the field is a constant and should not be differentiated by.
    outputs: list[int] = static_field(default_factory=lambda: [0])
    """Indeces of state vectors that are outputs. Defaults to `[0]`."""

    initial_state = jnp.zeros(2)
    n_inputs = "scalar"

    def f(self, x: Array) -> Array:
        """Constant-input part of the vector field.

        .. math: f(x) = [x_2, (-r x_2 - r_2 |x_2| x_2 - k x_1) / m]^T.

        """
        x1, x2 = x
        return jnp.array(
            [x2, (-self.r * x2 - self.r2 * jnp.abs(x2) * x2 - self.k * x1) / self.m]
        )

    def g(self, x: Array) -> Array:
        """Input-proportional part of the vector field.

        .. math: g(x) = [0, 1 / m]^T.

        """
        return jnp.array([0.0, 1.0 / self.m])

    def h(self, x: Array) -> Array:
        """Output function.

        .. math: y = h(x) = {x_j | j ∈ outputs}.

        """
        return x[jnp.array(self.outputs)]


class Sastry9_9(AbstractControlAffine):
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

    # The values of parameters can be constrained by initializing them with the
    # `non_negative_field` and `boxed_field` functions
    alpha: float = boxed_field(0.0, jnp.inf, default=0.0)
    beta: float = boxed_field(0.0, jnp.inf, default=0.0)
    gamma: float = boxed_field(0.0, jnp.inf, default=0.0)
    delta: float = non_negative_field(default=0.0)  # same as boxed_field(0, jnp.inf)

    initial_state = jnp.ones(2) * 0.5

    # Systems without inputs should set n_inputs to zero.
    n_inputs = 0

    def vector_field(self, x, u=None, t=None):
        x, y = x
        return jnp.array(
            [self.alpha * x - self.beta * x * y, self.delta * x * y - self.gamma * y]
        )


# We can also subclass already defined systems to further change their behaviour.
class LotkaVolterraWithTrainableInitialState(LotkaVolterra):
    # We can release parameter constraints with `field`. This will remove
    # the metadata on the corresponding field, indcating that this parameter is
    # unconstrained.
    alpha: float = field(default=1.0)

    # In constrast, the following line will not change the constraint on beta parameter,
    # only its default value, as the metadata of the field is unchanged.
    beta = 1.0

    # Here we redeclare the initial_state field to be trainable. When default values
    # with the *_field functions are set to mutable values (which funnily includes
    # jax.Array), one must use the `default_factory` argument.
    initial_state: Array = field(default_factory=lambda: jnp.ones(2) * 0.5)
