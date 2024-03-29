# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import typing


project = "Dynax"
copyright = "2023, Franz M. Heuchel"
author = "Franz M. Heuchel"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.aafig",
    #    "sphinx_autodoc_typehints",
    "nbsphinx",
]

bibtex_bibfiles = ["bibliography.bib"]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Scan documents for autosummary directives and generate stub pages for each.
autosummary_generate = True
autodoc_member_order = "bysource"

# List of autodoc directive flags that should be automatically applied
# to all autodoc directives.
autodoc_default_options = {
    "members": True,
    "inherited-members": False,
}

autoclass_content = "both"
autodoc_typehints = "signature"
typehints_use_signature = True


napoleon_include_init_with_doc = True
napoleon_preprocess_types = True
napoleon_attr_annotations = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Short type docs for jaxtyping's types
# https://github.com/patrick-kidger/pytkdocs_tweaks/blob/2a7ce453e315f526d792f689e61d56ecaa4ab000/pytkdocs_tweaks/__init__.py#L283
typing.GENERATING_DOCUMENTATION = True  # pyright: ignore
