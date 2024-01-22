"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

# -- Path setup --------------------------------------------------------------
import os
import warnings

import plotly.io as pio
import skfolio
from plotly.io._sg_scraper import plotly_sg_scraper
from sphinx_gallery.sorting import FileNameSortKey

# Configure plotly to integrate its output into the HTML pages generated by
# sphinx-gallery.
pio.renderers.default = "sphinx_gallery_png"  # "sphinx_gallery"

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=(
        "Values in x were outside bounds during a minimize step, clipping to bounds"
    ),
)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# -- Project information -----------------------------------------------------

project = "skfolio"
copyright = "2023, skfolio developers (BSD License)"
author = "Hugo Delatte"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx_design",
    "sphinx_copybutton",
    "matplotlib.sphinxext.plot_directive",
    "numpydoc",
    "sphinx_togglebutton",
    "sphinx_favicon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.imgconverter",
    "sphinx_gallery.gen_gallery",
    "sphinx-prompt",
    "sphinx.ext.mathjax",
    "sphinxext.opengraph",
    "sphinx_sitemap",
    "sphinx.ext.githubpages",
]

# Produce `plot::` directives for examples that contain `import matplotlib` or
# `from matplotlib import`.
numpydoc_use_plots = True

# Options for the `::plot` directive:
# https://matplotlib.org/stable/api/sphinxext_plot_directive_api.html
plot_formats = ["png"]
plot_include_source = True
plot_html_show_formats = False
plot_html_show_source_link = False

autodoc_default_options = {"members": True, "inherited-members": True}

# Don't show type hint in functions and classes
autodoc_typehints = "none"

# If false, no module index is generated.
html_domain_indices = False

# If false, no index is generated.
html_use_index = False

# If false, no module index is generated.
latex_domain_indices = False

# this is needed to remove warnings on the missing methods docstrings.
# see https://github.com/numpy/numpydoc/issues/69
numpydoc_class_members_toctree = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# The reST default role (used for this markup: `text`) to use for all
# documents.
default_role = "literal"

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# -- sphinxext-opengraph ----------------------------------------------------

ogp_site_url = "https://skfolio.org/"
ogp_site_name = "skfolio"
ogp_image = "https://skfolio.org/_images/expo.jpg"
ogp_enable_meta_description = True

# -- autosummary -------------------------------------------------------------

autosummary_generate = True

# -- sphinx_sitemap -------------------------------------------------------------
html_baseurl = "https://skfolio.org/"
sitemap_url_scheme = "{link}"

# -- Internationalization ----------------------------------------------------

# specifying the natural language populates some key tags
language = "en"

# -- MyST options ------------------------------------------------------------

# This allows us to use ::: to denote directives, useful for admonitions
myst_enable_extensions = ["colon_fence", "linkify", "substitution"]
myst_heading_anchors = 2
myst_substitutions = {"rtd": "[Read the Docs](https://readthedocs.org/)"}

# -- sphinx-favicons ------------------------------------------------------------
favicons = [
    {
        "rel": "shortcut icon",
        "type": "image/svg+xml",
        "sizes": "any",
        "href": "favicon.svg",
    },
    {
        "rel": "icon",
        "type": "image/svg+xml",
        "sizes": "any",
        "href": "favicon.svg",
    },
    {
        "rel": "icon",
        "type": "image/png",
        "sizes": "144x144",
        "href": "favicon.png",
    },
]

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_sourcelink_suffix = ""

# Define the version we use for matching in the version switcher.
# For local development, infer the version to match from the package.
release = skfolio.__version__
version_match = "v" + release

html_theme_options = {
    "pygment_light_style": "friendly",  # "friendly",
    "pygment_dark_style": "dracula",  # "monokai", # dracula highlight print
    "header_links_before_dropdown": 4,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/skfolio/skfolio",
            "icon": "fa-brands fa-github",
        },
    ],
    "logo": {
        "text": "skfolio",
        "alt_text": "skfolio documentation - Home",
        "image_light": "_static/favicon.svg",
        "image_dark": "_static/favicon.svg",
    },
    # "use_edit_page_button": True,
    "show_toc_level": 1,
    "navbar_align": (
        "left"
    ),  # [left, content, right] For testing that the navbar items align properly
    "announcement": """<div class="sidebar-message">
    If you'd like to contribute,
    <a href="https://github.com/skfolio/skfolio">check out our GitHub repository.</a>
    Your contributions are welcome!</div>""",
    "secondary_sidebar_items": [],  # No secondary sidebar due to bug with plotly
}

html_sidebars = {
    "auto_examples/*/*": [],  # no primary sidebar
    # "examples/persistent-search-field": ["search-field"],
}

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False

html_context = {
    "github_user": "skfolio",
    "github_repo": "skfolio",
    "github_version": "main",
    "doc_path": "docs",
    "default_mode": "dark",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
# html_js_files = ["custom-icon.js"]
# todo_include_todos = True

# -- gallery  ----------------------------------------------------------------

image_scrapers = (
    "matplotlib",
    plotly_sg_scraper,
)


class FileNameNumberSortKey(FileNameSortKey):
    """Sort examples in src_dir by file name number.

    Parameters
    ----------
    src_dir : str
        The source directory.
    """

    def __call__(self, filename):
        # filename="plot_10_tracking_error.py"
        return float(filename.split("_")[1])


sphinx_gallery_conf = {
    "doc_module": "skfolio",
    "backreferences_dir": os.path.join("modules", "generated"),
    "show_memory": False,
    "reference_url": {
        "skfolio": None,
    },
    "examples_dirs": ["../examples"],
    "gallery_dirs": ["auto_examples"],
    "within_subsection_order": FileNameNumberSortKey,
    "image_scrapers": image_scrapers,
    # avoid generating too many cross links
    "inspect_global_variables": False,
    "remove_config_comments": True,
    "plot_gallery": "True",
    "binder": {
        "org": "skfolio",
        "repo": "skfolio",
        "branch": "gh-pages",
        "binderhub_url": "https://mybinder.org",
        "dependencies": "./binder/requirements.txt",
        "use_jupyter_lab": True,
    },
    # 'compress_images': ('images', 'thumbnails'),
    # 'promote_jupyter_magic': False,
    # 'junit': os.path.join('sphinx-gallery', 'junit-results.xml'),
    # # capture raw HTML or, if not present, __repr__ of last expression in
    # # each code block
    # 'capture_repr': ('_repr_html_', '__repr__'),
    # 'matplotlib_animations': True,
    # 'image_srcset': ["2x"],
    # 'nested_sections': False,
    # 'show_api_usage': True,
}
