# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from pathlib import Path
from datetime import datetime
import os

# -- Path setup --------------------------------------------------------------
import sys

from sphinx.application import Sphinx

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, os.path.abspath("_ext"))

import pygpcca  # noqa: E402

needs_sphinx = "5"
# -- Project information -----------------------------------------------------

project = "pyGPCCA"
author = pygpcca.__author__
copyright = f"{datetime.now():%Y}, {author}"  # noqa: A001

# The full version, including alpha/beta/rc tags
master_doc = "index"
release = "main"
version = f"main ({pygpcca.__version__})"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "typed_returns",
    "myst_nb",
]
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["build", "**.ipynb_checkpoints"]

source_suffix = {".rst": "restructuredtext", ".ipynb": "myst-nb"}
add_function_parentheses = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
autosummary_generate = True
autodoc_member_order = "alphabetical"
autodoc_typehints = "signature"
autodoc_docstring_signature = True
autodoc_default_flags = ["inherited-members", "members"]
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True
napoleon_use_param = True
napoleon_custom_sections = [("Params", "Parameters"), ("Credits", "References")]
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
todo_include_todos = False

# myst-nb
myst_heading_anchors = 2
nb_execution_mode = "off"
nb_mime_priority_overrides = [("spelling", "text/plain", 0)]
myst_enable_extensions = [
    "colon_fence",
    "amsmath",
    "dollarmath",
]

# spelling
spelling_lang = "en_US"
spelling_warning = True
spelling_word_list_filename = "spelling_wordlist.txt"
spelling_add_pypi_package_names = True
# see: https://pyenchant.github.io/pyenchant/api/enchant.tokenize.html
spelling_filters = ["enchant.tokenize.URLFilter", "enchant.tokenize.EmailFilter"]

# linkcheck
linkcheck_anchors = False  # problem with specifying lines on GitHub in `acknowledgments.rst`
linkcheck_ignore = [
    "https://doi.org/10.1021/acs.jctc.8b00079",
    "https://pubs.acs.org/doi/abs/10.1021/acs.jctc.8b00079",
    "https://doi.org/10.1063/1.5064530",
]
linkcheck_report_timeouts_as_broken = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {"navigation_depth": 4, "logo_only": True}
html_show_sphinx = False
html_use_smartypants = True
pygments_style = "sphinx"


def setup(app: Sphinx) -> None:
    app.add_css_file("css/custom.css")
