# TODO: leave out or make clear somewhere
import jax as _jax

from .derivative import lie_derivative
from .estimation import fit_csd_matching, fit_least_squares
from .evolution import Flow, Map
from .interpolation import spline_it
from .system import (
    ControlAffine,
    DynamicalSystem,
    DynamicStateFeedbackSystem,
    FeedbackSystem,
    LinearSystem,
    SeriesSystem,
    StaticStateFeedbackSystem,
)


_jax.config.update("jax_enable_x64", True)
