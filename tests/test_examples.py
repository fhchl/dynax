import runpy
from pathlib import Path

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor


example_dir = Path(__file__, "..", "..", "examples").resolve()
examples = [str(p) for p in example_dir.glob("*.py")]
notebooks = [str(p) for p in example_dir.resolve().glob("*.ipynb")]


@pytest.mark.slow
@pytest.mark.parametrize("example", examples, ids=lambda x: Path(x).name)
def test_examples_run_without_error(example):
    runpy.run_path(example)


@pytest.mark.slow
@pytest.mark.parametrize("notebook", notebooks, ids=lambda x: Path(x).name)
def test_notebooks_dont_change(notebook):
    with open(notebook) as f:
        nb = nbformat.read(f, as_version=4)
        try:
            ExecutePreprocessor(timeout=60).preprocess(nb)
        except Exception as e:
            raise Exception(f"Running the notebook {notebook} failed") from e
