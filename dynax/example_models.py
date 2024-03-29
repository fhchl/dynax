import jax.numpy as jnp

from .system import (
    boxed_field,
    ControlAffine,
    DynamicalSystem,
    non_negative_field,
    static_field,
)


class PlasticFlowLinElastic(DynamicalSystem):
    kappa: float = non_negative_field()
    alpha: float
    sigma_0: float

    initial_state = jnp.zeros(2)
    n_inputs = "scalar"

    def vector_field(self, x, u, t=None):
        S, E = x
        return jnp.array(
            [
                (self.kappa + self.alpha) * u
                - self.alpha / self.sigma_0 * (S - self.kappa * E) * jnp.abs(u),
                u,
            ]
        )

    def output(self, x, u=None, t=None):
        S, E = x
        return S


class SpringMassDamper(DynamicalSystem):
    """Forced second-order linear spring-mass-damper system.

    .. math:: m x'' + r x' + k x = u.

    """

    m: float
    r: float
    k: float

    initial_state = jnp.zeros(2)
    n_inputs = "scalar"

    def vector_field(self, x, u, t=None):
        x1, x2 = x
        return jnp.array([x2, (u - self.r * x2 - self.k * x1) / self.m])


class NonlinearDrag(ControlAffine):
    """Spring-mass-damper system with nonlin drag.

    .. math:: m x'' +  r x' + r2 x'|x'| + k x = u.

    """

    r: float
    r2: float
    k: float
    m: float
    outputs: list[int] = static_field(default_factory=lambda: [0])

    initial_state = jnp.zeros(2)
    n_inputs = "scalar"

    def f(self, x):
        x1, x2 = x
        return jnp.array(
            [x2, (-self.r * x2 - self.r2 * jnp.abs(x2) * x2 - self.k * x1) / self.m]
        )

    def g(self, x):
        return jnp.array([0.0, 1.0 / self.m])

    def h(self, x):
        return x[jnp.array(self.outputs)]


class Sastry9_9(ControlAffine):
    """Sastry Example 9.9"""

    initial_state = jnp.zeros(3)
    n_inputs = "scalar"

    def f(self, x):
        return jnp.array([0.0, x[0] + x[1] ** 2, x[0] - x[1]])

    def g(self, x):
        return jnp.array([jnp.exp(x[1]), jnp.exp(x[1]), 0.0])

    def h(self, x):
        return x[2]


class LotkaVolterra(DynamicalSystem):
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


class SpringMassWithBoucWenHysteresis(DynamicalSystem):
    """https://en.wikipedia.org/wiki/Bouc%E2%80%93Wen_model_of_hysteresis"""

    m: float = non_negative_field()  # kg
    r: float = non_negative_field()  # Ns/m
    ki: float = non_negative_field()  # N/m
    gamma: float = non_negative_field()
    n: float = non_negative_field(min_val=1.0)
    a: float = boxed_field(0.0, 1.0)

    initial_state = jnp.zeros(3)

    def vector_field(self, x, u=None, t=None):
        if u is None:
            u = 0
        f = u
        u, du, z = x
        # remove parameter redundancies
        A = 1
        beta = A - self.gamma
        # restoring force with hysteresis
        F = self.a * self.ki * u + (1 - self.a) * self.ki * z
        # shape control function
        psi = beta * jnp.sign(z * du) + self.gamma
        return jnp.array(
            [
                du,
                (f - self.r * du - F) / self.m,
                du * (A - psi * jnp.power(jnp.abs(z), self.n)),
            ]
        )
