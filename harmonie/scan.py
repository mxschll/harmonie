"""Filesystem scanning for audio files."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Iterator, Optional

AUDIO_EXTENSIONS = frozenset(
    {
        ".flac", ".mp3", ".wav", ".ogg", ".oga", ".m4a", ".aac",
        ".aiff", ".aif", ".wma", ".opus", ".alac",
    }
)


def is_audio_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS


def iter_audio_files(roots: Iterable[Path]) -> Iterator[Path]:
    """Yield audio files under each root, recursively. Roots may be files
    or directories. Hidden directories are skipped, symlinks are not
    followed, and results are de-duplicated by realpath.
    """
    seen: set[str] = set()
    for root in roots:
        root = Path(root).expanduser()
        if not root.exists():
            continue
        if root.is_file():
            if is_audio_file(root):
                key = os.path.realpath(root)
                if key not in seen:
                    seen.add(key)
                    yield root
            continue
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for name in filenames:
                p = Path(dirpath) / name
                if is_audio_file(p):
                    key = os.path.realpath(p)
                    if key not in seen:
                        seen.add(key)
                        yield p


def split_library_path(
    path: str, libraries: Iterable[Path]
) -> tuple[Optional[str], Optional[str]]:
    """Find which configured library root contains ``path``.

    Returns ``(library_root, relative_path)`` as resolved absolute strings,
    or ``(None, None)`` if the path isn't under any of ``libraries``.

    The first matching library wins. Both inputs are resolved to absolute
    paths before comparison.
    """
    try:
        target = Path(path).expanduser().resolve()
    except Exception:
        return None, None
    for root in libraries:
        try:
            root_resolved = Path(root).expanduser().resolve()
        except Exception:
            continue
        try:
            rel = target.relative_to(root_resolved)
        except ValueError:
            continue
        return str(root_resolved), str(rel)
    return None, None

