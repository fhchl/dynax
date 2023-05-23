import importlib

import jax as jax
from equinox import static_field as static_field

from .derivative import lie_derivative
from .estimation import (
    fit_csd_matching,
    fit_least_squares,
    fit_multiple_shooting,
    non_negative_field,
)
from .evolution import AbstractEvolution, Flow, Map
from .interpolation import spline_it
from .linearize import input_output_linearize, relative_degree
from .system import (
    ControlAffine,
    DynamicalSystem,
    DynamicStateFeedbackSystem,
    FeedbackSystem,
    LinearSystem,
    SeriesSystem,
    StaticStateFeedbackSystem,
)


# TODO: leave out or make clear somewhere
print("Setting jax_enable_x64 to True.")
jax.config.update("jax_enable_x64", True)

__version__ = importlib.metadata.version("equinox")

__all__ = [
    input_output_linearize,
    relative_degree,
    static_field,
    lie_derivative,
    fit_csd_matching,
    fit_least_squares,
    fit_multiple_shooting,
    non_negative_field,
    AbstractEvolution,
    Flow,
    Map,
    spline_it,
    ControlAffine,
    DynamicalSystem,
    DynamicStateFeedbackSystem,
    FeedbackSystem,
    LinearSystem,
    SeriesSystem,
    StaticStateFeedbackSystem,
]
