import pathlib
import runpy

import pytest


examples = pathlib.Path(__file__, "..", "..", "examples").resolve().glob("*.py")


@pytest.mark.slow
@pytest.mark.parametrize("examples", examples)
def test_examples_run_without_error(examples):
    runpy.run_path(examples)
