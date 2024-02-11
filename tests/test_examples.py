import pathlib
import runpy

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor


examples = pathlib.Path(__file__, "..", "..", "examples").resolve().glob("*.py")
notebooks = pathlib.Path(__file__, "..", "..", "examples").resolve().glob("*.ipynb")


@pytest.mark.slow
@pytest.mark.parametrize("example", examples, ids=lambda x: str(x.name))
def test_examples_run_without_error(example):
    runpy.run_path(example)


@pytest.mark.slow
@pytest.mark.parametrize("notebook", notebooks, ids=lambda x: str(x.name))
def test_notebooks_dont_change(notebook):
    with open(notebook) as f:
        nb = nbformat.read(f, as_version=4)
        try:
            ExecutePreprocessor(timeout=60).preprocess(nb)
        except Exception as e:
            raise Exception(f"Running the notebook {notebook} failed") from e
