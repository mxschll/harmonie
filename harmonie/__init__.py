"""harmonie: audio similarity service."""

# Version is generated at build time by setuptools-scm into _version.py.
# Three layers of fallback so we never crash on import:
#   1. The build-time generated _version.py (normal pip-installed case).
#   2. importlib.metadata (works inside an installed wheel even if step 1 missed).
#   3. A literal fallback for editable / source checkouts without git history.
try:
    from harmonie._version import (
        version as __version__,  # type: ignore[import-not-found]
    )
except ImportError:
    try:
        from importlib.metadata import version as _pkg_version

        __version__ = _pkg_version("harmonie")
    except Exception:
        __version__ = "0.0.0+unknown"
