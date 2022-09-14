from dynax.ad import *
import numpy.testing as npt
import jax
from test_linearize import Sastry9_9


def test_lie_derivative():
  sys = Sastry9_9()
  f = sys.f
  g = sys.g
  h = sys.h
  
  np.random.seed(0)
  xs = np.random.normal(size=(10, 3))
  for x in xs:
    x1, x2, x3 = x
    npt.assert_allclose(lie_derivative(f, h, n=1)(x), x1 - x2)
    npt.assert_allclose(lie_derivative(f, h, n=2)(x), -x1 - x2**2)
    npt.assert_allclose(lie_derivative(f, h, n=3)(x), -2*x2*(x1+x2**2))
    npt.assert_allclose(lie_derivative(g, h, n=1)(x), 0)
    npt.assert_allclose(lie_derivative(g, lie_derivative(f, h, n=1))(x), 0)
    npt.assert_allclose(lie_derivative(g, lie_derivative(f, h, n=2))(x),
                        -(1+2*x2)*np.exp(x2), rtol=1e-6)


def test_lie_derivative2():
  sys = Sastry9_9()
  f = sys.f
  g = sys.g
  h = sys.h
  
  np.random.seed(0)
  xs = np.random.normal(size=(10, 3))
  tol = dict(atol=1e-8, rtol=1e-6)
  
  for x in xs:
    x1, x2, _ = x
    npt.assert_allclose(lie_derivative2(f, h, n=3)(x),
                        [h(x), x1 - x2, -x1 - x2**2, -2*x2*(x1+x2**2)], **tol)
    npt.assert_allclose(lie_derivative2(g, h, n=1)(x)[-1], 0, **tol)
    npt.assert_allclose(jax.jit(lie_derivative2(g, h, n=1))(x)[-1], 0, **tol)
    #npt.assert_allclose(lie_derivative2(g, lie_derivative2(f, h, n=1))(x)[-1], 0)
    #npt.assert_allclose(lie_derivative2(g, lie_derivative2(f, h, n=2))(x),
    #                    -(1+2*x2)*np.exp(x2), rtol=1e-6)


#test_lie_derivative()
test_lie_derivative2()

import time
from benchmarks import benchmark
import jax.numpy as jnp

def test_lie_derivative_speed():
  sys = Sastry9_9()
  f = sys.f
  h = sys.h
  
  np.random.seed(0)
  xs = jnp.array(np.random.normal(size=(10, 3)))
  fun = jax.jit(lie_derivative(f, h, n=10))
  benchmark(lambda: fun(jnp.array(np.random.normal(size=3))), iters=10)
  # FIXME: problem might be that lie_derivative2 only works with scalars?
  fun2 = jax.jit(lie_derivative2(f, h, n=10))
  benchmark(lambda: fun2(jnp.array(np.random.normal(size=3))), iters=10)

test_lie_derivative_speed()

