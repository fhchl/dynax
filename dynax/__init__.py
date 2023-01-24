import jax

from .estimation import fit_csd_matching, fit_least_squares
from .interpolation import spline_it
from .system import (
    ControlAffine,
    DiscreteForwardModel,
    DynamicalSystem,
    DynamicStateFeedbackSystem,
    FeedbackSystem,
    ForwardModel,
    LinearSystem,
    SeriesSystem,
    StaticStateFeedbackSystem,
)


# TODO: leave out or make clear somewhere
jax.config.update("jax_enable_x64", True)
