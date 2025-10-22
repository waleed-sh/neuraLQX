# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'neuraLQX'
copyright = '2025, The neuraLQX Authors'
author = 'neuraLQX'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx_design",
    "myst_parser",             # Enables Markdown support
    "sphinx.ext.autodoc",      # For automatic API docs later
    "sphinx.ext.napoleon",     # Supports Google/NumPy docstrings
    "sphinx_copybutton",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'shibuya'
html_static_path = ['_static']
html_theme_options = {
    "light_logo": "_static/logo_banner.png",
    "dark_logo": "_static/logo_banner.png",
    "logo_target": "https://neuralqx.readthedocs.io/en/latest/",
    "accent_color": "teal",
    "color_mode": "light",
    "page_layout": "default",
    "announcement": "This package is still underdevelopment. Stay tuned for the release!",
}

# MyST extensions
myst_enable_extensions = [
    "colon_fence",  # allows ::: blocks
    "deflist",
    "html_admonition",
    "html_image",
]