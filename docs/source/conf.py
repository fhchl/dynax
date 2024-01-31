# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Dynax"
copyright = "2023, Franz M. Heuchel"
author = "Franz M. Heuchel"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
    "sphinx_autodoc_typehints",
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
# autodoc_typehints = "signature"
# typehints_use_signature = "True"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
