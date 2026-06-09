# Configuration file for the Sphinx documentation builder.
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'Emergo'
copyright = '2026, Emergo Contributors'
author = 'Emergo Contributors'
release = '0.2.0'
version = '0.2.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'alabaster'
html_static_path = ['_static']
html_theme_options = {
    'description': 'A self-organizing intelligence engine for multi-agent coordination.',
    'github_user': 'markhamcarter220-sketch',
    'github_repo': 'Emergo',
    'github_button': True,
    'fixed_sidebar': True,
}

autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
    'special-members': '__init__',
}
autodoc_typehints = 'description'
napoleon_google_docstring = True
napoleon_numpy_docstring = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
}

autosummary_generate = True
add_module_names = False
