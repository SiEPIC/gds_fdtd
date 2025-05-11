import os
import sys
sys.path.insert(0, os.path.abspath('../'))

# Project details
project = 'gds_fdtd'
author = 'Mustafa Hammood'
release = '0.3.6'

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # Google-style docstrings
    'sphinx.ext.viewcode',
    'sphinx_toggleprompt',  # For interactive prompts
    'sphinx_copybutton',    # Adds copy buttons to code blocks
]

# Theme
html_theme = 'furo'

# Toggle Light/Dark mode (built into furo)
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#3498db",
        "color-brand-content": "#2ecc71",
    },
    "dark_css_variables": {
        "color-brand-primary": "#9b59b6",
        "color-brand-content": "#e74c3c",
    },
}



# Paths
templates_path = ['_templates']
exclude_patterns = []
html_static_path = ['_static']
