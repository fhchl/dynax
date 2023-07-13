import importlib.metadata

import jax as jax
from equinox import static_field as static_field

from .derivative import lie_derivative as lie_derivative
from .estimation import (
    boxed_field as boxed_field,
    fit_csd_matching as fit_csd_matching,
    fit_least_squares as fit_least_squares,
    fit_multiple_shooting as fit_multiple_shooting,
    non_negative_field as non_negative_field,
)
from .evolution import AbstractEvolution as AbstractEvolution, Flow as Flow, Map as Map
from .interpolation import spline_it as spline_it
from .linearize import (
    input_output_linearize as input_output_linearize,
    relative_degree as relative_degree,
)
from .system import (
    ControlAffine as ControlAffine,
    DynamicalSystem as DynamicalSystem,
    DynamicStateFeedbackSystem as DynamicStateFeedbackSystem,
    FeedbackSystem as FeedbackSystem,
    LinearSystem as LinearSystem,
    SeriesSystem as SeriesSystem,
    StaticStateFeedbackSystem as StaticStateFeedbackSystem,
)
from .util import monkeypatch_pretty_print, pretty as pretty


# TODO: leave out or make clear somewhere
print("Setting jax_enable_x64 to True.")
jax.config.update("jax_enable_x64", True)

monkeypatch_pretty_print()

__version__ = importlib.metadata.version("dynax")
