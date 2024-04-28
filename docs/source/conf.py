import typing


html_theme = "sphinx_rtd_theme"
# html_static_path = ["_static"]

project = "Dynax"
copyright = "2023, Franz M. Heuchel"
author = "Franz M. Heuchel"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.aafig",
    "nbsphinx",
    "sphinx.ext.napoleon",
    # FIXME: sphinx_autodoc_typehints is not working together with autodoc_type_aliases,
    # see https://github.com/tox-dev/sphinx-autodoc-typehints/issues/284
    # For now, I will just use jaxtyping.ArrayLike in the docs. Sadly, that one does
    # not intersphinx-link to the docs.
    "sphinx_autodoc_typehints",
]

bibtex_bibfiles = ["bibliography.bib"]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_default_flags = ["show-inheritance", "members", "inherited-members"]
autodoc_default_options = {
    "members": True,
    "inherited-members": False,
    "show-inheritance": True,
}
autoclass_content = "both"
autodoc_typehints = "signature"
autodoc_preserve_defaults = True
autodoc_type_aliases = {
    a: a
    for a in [
        # "VectorFunc",
        # "ScalarFunc",
        "VectorField",
        "OutputFunc",
        # "ArrayLike"
    ]
}
# } | {
#     "jax.typing.ArrayLike": "jax.typing.ArrayLike",
#     "ArrayLike": "dynax.custom_types.ArrayLike",
#     "jax._src_basearray.ArrayLike": "ArrayLike",
# }

# TODO: I want to stop ArrayLike from exploiding, but above doesn't seem to work :/

# For sphinx_autodoc_typehints.
typehints_use_rtype = False
always_use_bars_union = True

napoleon_numpy_docstring = False
napoleon_google_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_special_with_doc = False
napoleon_preprocess_types = True
napoleon_attr_annotations = True
napoleon_use_rtype = False

# TODO: __init__ should not pop up in the docs
# TODO: remove dynax.evolution... path from docs
# TODO: intersphinx should pick up Module, Array and ArrayLike, but doesn't :(


intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "jaxtyping": ("https://docs.kidger.site/jaxtyping/", None),
    "diffrax": ("https://docs.kidger.site/diffrax/", None),
    "equinox": ("https://docs.kidger.site/equinox/", None),
    "optimistix": ("https://docs.kidger.site/optimistix/", None),
    "lineax": ("https://docs.kidger.site/lineax/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
}


# def autodoc_process_docstring(app, what, name, obj, options, lines):
#     for i in range(len(lines)):
#         if lines[i]
#         # # lines[i] = lines[i].replace("np.", "~numpy.")  # For shorter links
#         # lines[i] = lines[i].replace("F.", "torch.nn.functional.")
#         # lines[i] = lines[i].replace("List[", "~typing.List[")


# def setup(app):
#     app.connect("autodoc-process-docstring", autodoc_process_docstring)

# Short type docs for jaxtyping's types
# https://github.com/patrick-kidger/pytkdocs_tweaks/blob/2a7ce453e315f526d792f689e61d56ecaa4ab000/pytkdocs_tweaks/__init__.py#L283
typing.GENERATING_DOCUMENTATION = True  # pyright: ignore
