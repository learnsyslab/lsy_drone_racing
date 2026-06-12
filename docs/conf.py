# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os

# -- Project information -----------------------------------------------------

project = "lsy_drone_racing"
copyright = "2024, Learning Systems and Robotics Lab (LSY) @ TUM"
author = "Martin Schuck"

# The full version, including alpha/beta/rc tags
release = "0.1.0"
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.todo"]
add_module_names = False

# Autodoc config
autodoc_member_order = "bysource"

# Mock imports on ReadTheDocs that are not available with pip or that we do not want to install
autodoc_mock_imports = ["cflib", "crazyflow", "drone_models", "drone_estimators", "torch", "rclpy", "mujoco", "acados_template"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"
html_logo = "img/banner.jpeg"
html_favicon = "img/banner.jpeg"
html_theme_options = {
    "repository_url": "https://github.com/learnsyslab/lsy_drone_racing",
    "use_repository_button": True,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []
