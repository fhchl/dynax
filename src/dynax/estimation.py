from dataclasses import field, fields
from typing import Callable, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_flatten, tree_map, tree_structure
from scipy.optimize import least_squares
from scipy.optimize._optimize import MemoizeJac

from .custom_types import Array, PyTree
from .system import (DiscreteForwardModel, DynamicalSystem, ForwardModel,
                     spline_it)
from .util import value_and_jacfwd


def boxed_field(lower: float, upper: float, **kwargs):
  """Mark a dataclass field as having a box-constrained value."""
  try:
      metadata = dict(kwargs["metadata"])
  except KeyError:
      metadata = kwargs["metadata"] = {}
  if "constrained" in metadata:
      raise ValueError("Cannot use metadata if `constrained` already set.")
  metadata["constrained"] = ("boxed", (lower, upper))
  return field(**kwargs)


def non_negative_field(min_val=0., **kwargs):
  """Mark a dataclass field as having a non-negative value."""
  return boxed_field(lower=min_val, upper=np.inf, **kwargs)


# TODO: In the best case, this function should return PyTrees of the same form
#       as `self`. Right now however, the function is NOT recursive, so won't
#       work on e.g. ForwardModel.
def build_bounds(self: DynamicalSystem) -> Tuple[PyTree, PyTree]:
  """Build PyTrees of lower and upper bounds."""
  lower_bounds = []
  upper_bounds = []
  for field_ in fields(self):
    name = field_.name
    try:
      value = self.__dict__[name]
    except KeyError:
      continue
    # static parameters have no bounds
    if field_.metadata.get("static", False):
      continue
    # dynamic value has bounds
    elif bound := field_.metadata.get("constrained", False):
      kind, aux = bound
      if kind == "boxed":
        lower, upper = aux
        lower_bounds.append(tree_map(lambda _: lower, value))
        upper_bounds.append(tree_map(lambda _: upper, value))
      else:
        raise ValueError("Unknown bound type {kind}.")
    # dynamic value is unbounded
    else:
      lower_bounds.append(tree_map(lambda _: -np.inf, value))
      upper_bounds.append(tree_map(lambda _: np.inf, value))
  treedef = tree_structure(self)
  return (treedef.unflatten(tuple(lower_bounds)),
          treedef.unflatten(tuple(upper_bounds)))


def fit_ml(model: ForwardModel | DiscreteForwardModel,
           t: Array,
           y: Array,
           x0: Array,
           u: Callable[[float], Array] | Array | None = None
           ) -> ForwardModel | DiscreteForwardModel:
  """Fit forward model via maximum likelihood."""
  t = jnp.asarray(t)
  y = jnp.asarray(y)
  if (isinstance(model, ForwardModel) and u is not None and
      not isinstance(u, Callable)):
    u = spline_it(t, u)

  # TODO: if model or any sub tree has some ndarray leaves, this does not
  # completely flatten the arrays. But a flat array is needed for `least_squares`.
  # Same problem appears for the bounds.
  init_params, treedef = tree_flatten(model)
  std_y = np.std(y, axis=0)
  bounds = tuple(map(lambda x: tree_flatten(x)[0], build_bounds(model.system)))

  def residuals(params):
    model = treedef.unflatten(params)
    pred_y, _ = model(x0, t=t, u=u)
    res = ((y - pred_y)/std_y).reshape(-1)
    return res / np.sqrt(len(res))

  # compute primal and sensitivties in one forward pass
  fun = MemoizeJac(jax.jit(lambda x: value_and_jacfwd(residuals, x)))
  jac = fun.derivative
  # use instead:
  # - https://lmfit.github.io/lmfit-py/index.html
  # - https://github.com/dipolar-quantum-gases/jaxfit
  # - scipy.optimize.curve_fit
  res = least_squares(fun, init_params, bounds=bounds, jac=jac, x_scale='jac',
                      verbose=2)
  params = res.x
  return treedef.unflatten(params)