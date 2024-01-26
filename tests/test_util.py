import jax
import jax.numpy as jnp
import numpy.testing as npt

from dynax.util import ssmatrix


def test_ssmatrix():
    assert ssmatrix(0.) == jnp.array([[0.]])
    
    y = ssmatrix(jnp.array([]))
    assert y.shape == (0, 0)

    npt.assert_array_equal(ssmatrix(jnp.array([0., 1., 2.])),
                           jnp.array([[0., 1., 2.]]).T)
    
