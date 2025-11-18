

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import inspect
import os
import sys

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'neuraLQX'
copyright = '2025, The neuraLQX Authors - All Rights Reserved'
author = 'The neuraLQX Authors'

# change to nqx.__version__ when releasing
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx_design",
    # "myst_parser",
    "sphinx.ext.autodoc",      # For automatic API docs later
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "myst_nb",
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    # "sphinx.ext.linkcode",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.graphviz",
]

autodoc_docstring_signature = True
autodoc_inherit_docstrings = True
allow_inherited = True
autosummary_generate = True
napoleon_preprocess_types = True
napoleon_attr_annotations = True


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', "**.ipynb_checkpoints", "README.md"]
source_suffix = [".rst", ".ipynb", ".md"]

# MyST extensions
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence",
    "html_admonition",
    "substitution",
    "deflist",
    "fieldlist",
    "tasklist",
    "colon_fence",  # allows ::: blocks
    "deflist",
    "html_admonition",
    "html_image",
]

myst_update_mathjax = False
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
myst_heading_anchors = 2
autosectionlabel_maxdepth = 1

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "netket": ("https://netket.readthedocs.io/en/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    # "flax": ("https://flax.readthedocs.io/en/latest/", None),
    "flax": ("https://flax-linen.readthedocs.io/en/latest/", None),
}

html_theme = 'shibuya'
html_static_path = ['_static']
html_title = "neuraLQX"
html_logo = "_static/logo_banner_tbg.png"
# html_favicon = "_static/favicon.ico"

html_theme_options = {
    "light_logo": "_static/logo_banner_tbg.png",
    "dark_logo": "_static/logo_banner_tbg.png",
    "logo_target": "https://neuralqx.readthedocs.io/en/latest/",
    "accent_color": "teal",
    "color_mode": "light",
    "page_layout": "default",
    "announcement": "This package is still underdevelopment. Stay tuned for the release!",
    "globaltoc_expand_depth": 2,
    # "nav_links": [
    #     {
    #         "title": "Examples",
    #         "url": "writing",
    #         "children": [
    #             {
    #                 "title": "Admonitions",
    #                 "url": "writing/admonition",
    #             },
    #             {
    #                 "title": "Code Blocks",
    #                 "url": "writing/code",
    #             },
    #             {
    #                 "title": "Autodoc",
    #                 "url": "writing/api",
    #             },
    #         ]
    #     },
    # ]
}

nb_execution_mode = "off"
nb_execution_allow_errors = False

nb_render_markdown_format = "myst"
nb_merge_streams = True

