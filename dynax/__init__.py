import importlib.metadata

import jax as _jax

from .derivative import lie_derivative as lie_derivative
from .estimation import (
    fit_csd_matching as fit_csd_matching,
    fit_least_squares as fit_least_squares,
    fit_multiple_shooting as fit_multiple_shooting,
    transfer_function as transfer_function,
)
from .evolution import AbstractEvolution as AbstractEvolution, Flow as Flow, Map as Map
from .interpolation import spline_it as spline_it
from .linearize import (
    discrete_input_output_linearize as discrete_input_output_linearize,
    discrete_relative_degree as discrete_relative_degree,
    DiscreteLinearizingSystem as DiscreteLinearizingSystem,
    input_output_linearize as input_output_linearize,
    LinearizingSystem as LinearizingSystem,
    relative_degree as relative_degree,
)
from .system import (
    AbstractControlAffine as AbstractControlAffine,
    AbstractSystem as AbstractSystem,
    boxed_field as boxed_field,
    DynamicStateFeedbackSystem as DynamicStateFeedbackSystem,
    FeedbackSystem as FeedbackSystem,
    field as field,
    LinearSystem as LinearSystem,
    non_negative_field as non_negative_field,
    SeriesSystem as SeriesSystem,
    static_field as static_field,
    StaticStateFeedbackSystem as StaticStateFeedbackSystem,
)
from .util import _monkeypatch_pretty_print, pretty as pretty


# TODO: leave out or make clear somewhere
print("Setting jax_enable_x64 to True.")
_jax.config.update("jax_enable_x64", True)

_monkeypatch_pretty_print()

__version__ = importlib.metadata.version("dynax")
