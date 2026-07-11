import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath("../src"))

# The example gallery: the executed notebooks live under ../examples/ (their
# source of truth), which is outside the Sphinx source tree. Copy them into a
# build-time _notebooks/ dir so myst-nb can render them in the docs. The copies
# are a build artifact (gitignored); the committed .ipynb outputs are shown
# as-is (nb_execution_mode = "off"), so nothing is re-run during the docs build.
_HERE = Path(__file__).parent
_NB_DIR = _HERE / "_notebooks"
shutil.rmtree(_NB_DIR, ignore_errors=True)
_NB_DIR.mkdir(exist_ok=True)
for _nb in sorted((_HERE.parent / "examples").glob("[0-9]*/[0-9]*.ipynb")):
    shutil.copy2(_nb, _NB_DIR / _nb.name)

# Project details
project = "gds_fdtd"
author = "Mustafa Hammood"
try:
    from importlib.metadata import version as _pkg_version

    release = _pkg_version("gds_fdtd")
except Exception:  # pragma: no cover - docs built without install
    release = "0.0.0+unknown"

# Extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",  # Google-style docstrings
    "sphinx.ext.viewcode",
    "sphinx_toggleprompt",  # For interactive prompts
    "sphinx_copybutton",  # Adds copy buttons to code blocks
    "myst_nb",  # Markdown pages + executed-notebook gallery (supersedes myst_parser)
]

myst_enable_extensions = ["colon_fence"]

# Show the committed notebook outputs as-is; never re-execute during the build.
nb_execution_mode = "off"
# Long single-cell outputs (beamz raster logs, etc.) shouldn't abort the build.
nb_output_stderr = "remove"
nb_merge_streams = True

autosummary_generate = True

# engine packages are optional extras; docs build without them installed
autodoc_mock_imports = ["tidy3d", "lumapi", "beamz", "gdsfactory", "prefab", "SiEPIC", "pya", "jax"]

# Theme
html_theme = "furo"

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
templates_path = ["_templates"]
exclude_patterns = []
html_static_path = ["_static"]
