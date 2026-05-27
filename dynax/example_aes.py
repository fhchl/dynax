import jax.numpy as jnp
import numpy as np

from dynax.custom_types import Array
from dynax.system import (
    AbstractControlAffine,
    non_negative_field,
)


class NonlinearDrag(AbstractControlAffine):
    """Forced spring-mass-damper system with nonlin drag.

    .. math:: m x'' +  r x' + r_2 x'|x'| + k x = u.

    """

    r: Array = non_negative_field()
    r2: Array = non_negative_field()
    k: Array = non_negative_field()
    m: Array = non_negative_field()

    initial_state = np.zeros(2)
    n_inputs = "scalar"

    def f(self, x):
        d, v = x
        return jnp.array(
            [v, (-self.r * v - self.r2 * jnp.abs(v) * v - self.k * d) / self.m]
        )

    def g(self, x):
        return jnp.array([0.0, 1.0 / self.m])

    def h(self, x):
        d, _ = x
        return d
