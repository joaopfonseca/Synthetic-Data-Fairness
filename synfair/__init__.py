"""Toolbox to monitor bias and quality of synthetic data.

``synfair`` is a library containing the implementation of various tools
and utilities to assess the quality of synthetic data.

Subpackages
-----------
datasets
    Module which contains code to download, transform and simulate various datasets.
"""

import sys

try:
    # This variable is injected in the __builtins__ by the build
    # process. It is used to enable importing subpackages of synfair when
    # the binaries are not built
    # mypy error: Cannot determine type of '__SYNFAIR_SETUP__'
    __SYNFAIR_SETUP__  # type: ignore
except NameError:
    __SYNFAIR_SETUP__ = False

if __SYNFAIR_SETUP__:
    sys.stderr.write("Partial import of synfair during the build process.\n")
    # We are not importing the rest of synfair during the build
    # process, as it may not be compiled yet
else:
    from . import datasets
    from ._version import __version__
    from .utils._show_versions import show_versions

    __all__ = [
        "datasets",
        # Non-modules:
        "show_versions",
        "__version__",
    ]
