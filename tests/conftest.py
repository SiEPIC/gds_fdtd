"""Suite-wide test configuration.

Force the non-interactive matplotlib backend before anything imports pyplot:
Windows CI runners ship a broken Tk (tcl_findLibrary errors), and no test
should ever open a GUI window anyway.
"""

import matplotlib

matplotlib.use("Agg", force=True)
