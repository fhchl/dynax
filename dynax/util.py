import functools
import jax
import jax.numpy as jnp
import numpy as np

def value_and_jacfwd(f, x):
  """Create a function that evaluates both fun and its foward-mode jacobian.

  Only works on ndarrays, not pytrees.
  Source: https://github.com/google/jax/pull/762#issuecomment-1002267121
  """
  pushfwd = functools.partial(jax.jvp, f, (x,))
  basis = jnp.eye(x.size, dtype=x.dtype)
  y, jac = jax.vmap(pushfwd, out_axes=(None, 1))((basis,))
  return y, jac

def value_and_jacrev(f, x):
  """Create a function that evaluates both fun and its reverse-mode jacobian.

  Only works on ndarrays, not pytrees.
  Source: https://github.com/google/jax/pull/762#issuecomment-1002267121
  """
  y, pullback = jax.vjp(f, x)
  basis = jnp.eye(y.size, dtype=y.dtype)
  jac = jax.vmap(pullback)(basis)
  return y, jac

class MemoizeJac:
  """ Decorator that caches the return values of a function returning `(fun, grad)`
      each time it is called.

  Source: https://github.com/scipy/scipy/blob/85895a2fdfed853801846b56c9f1418886e2ccc2/scipy/optimize/_optimize.py#L57
  """

  def __init__(self, fun):
    self.fun = fun
    self.jac = None
    self._value = None
    self.x = None

  def _compute_if_needed(self, x, *args, der=False):
    if not np.all(x == self.x) or self._value is None or self.jac is None:
      if der:
        print("Cache missed.")
      self.x = np.asarray(x).copy()
      fg = self.fun(x, *args)
      self.jac = fg[1]
      self._value = fg[0]

  def __call__(self, x, *args):
    """ returns the the function value """
    self._compute_if_needed(x, *args)
    return self._value

  def derivative(self, x, *args):
    self._compute_if_needed(x, *args, der=True)
    return self.jac