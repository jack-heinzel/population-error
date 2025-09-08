# Configuration file for the Sphinx documentation builder.

import os
import sys
import importlib.metadata as importlib_metadata

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "population-error"
author = "Jack Heinzel"

# Read the version from your installed package metadata
try:
    release = importlib_metadata.version(project)
except importlib_metadata.PackageNotFoundError:
    release = "0.0.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",       # Pull in docstrings automatically
    "sphinx.ext.napoleon",      # Support Google/NumPy style docstrings
    "sphinx.ext.viewcode",      # Add links to highlighted source code
    "sphinx.ext.githubpages",   # Create .nojekyll file for GitHub Pages
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
