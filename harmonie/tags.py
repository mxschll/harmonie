"""Audio tag extraction (ID3, Vorbis comments, MP4 atoms) via mutagen.

Tags are not "features" in the Essentia sense — they're file metadata set by
the user (or their tagger). We read them so external clients can match
harmonie tracks back to their own catalog without doing a full filesystem
walk: artist + album + title + track number is good enough for the long tail.

The extractor is defensive: any failure inside mutagen returns an empty
:class:`Tags`. Tag extraction must never abort a scan.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger("harmonie.tags")


@dataclass
class Tags:
    """Subset of audio tags useful for matching against external catalogs."""

    artist: Optional[str] = None
    album: Optional[str] = None
    title: Optional[str] = None
    track_number: Optional[int] = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _coerce_string(v: Any) -> Optional[str]:
    """Mutagen returns lists of str, ASF attributes, MP4FreeForm bytes, etc.
    Pull the first sensible string out."""
    if v is None:
        return None
    if isinstance(v, list):
        if not v:
            return None
        v = v[0]
    if isinstance(v, bytes):
        try:
            v = v.decode("utf-8", errors="replace")
        except Exception:
            return None
    s = str(v).strip()
    return s or None


def _parse_track_number(v: Any) -> Optional[int]:
    """``"5/12"`` -> 5, ``5`` -> 5, ``(5, 12)`` -> 5, garbage -> None."""
    if v is None:
        return None
    if isinstance(v, list):
        if not v:
            return None
        v = v[0]
    if isinstance(v, tuple):
        v = v[0] if v else None
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    if "/" in s:
        s = s.split("/", 1)[0].strip()
    try:
        return int(s)
    except ValueError:
        return None


def _first_value(tags: Any, keys: Iterable[str]) -> Any:
    """Return the first non-empty value from ``tags`` for any of ``keys``.

    Works with both dict-like containers (Vorbis comments, easy-id3) and the
    raw mutagen tag objects which behave like lists of frames.
    """
    if tags is None:
        return None
    for key in keys:
        try:
            v = tags.get(key) if hasattr(tags, "get") else tags[key]
        except (KeyError, TypeError):
            continue
        if v:
            return v
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_tags(path: Path) -> Tags:
    """Read tags from an audio file. Returns empty :class:`Tags` on any failure.

    Uses mutagen's "easy" wrappers where available so keys are normalised
    across formats (FLAC, MP3, M4A, OGG, etc.).
    """
    try:
        import mutagen  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover
        logger.warning("mutagen not installed; tag extraction disabled")
        return Tags()

    try:
        f = mutagen.File(str(path), easy=True)
    except Exception as e:  # mutagen has many exotic exception classes
        logger.debug("mutagen failed on %s: %s", path, e)
        return Tags()
    if f is None:
        return Tags()

    tag_obj = getattr(f, "tags", None)
    if tag_obj is None:
        return Tags()

    artist_raw = _first_value(tag_obj, ("artist", "albumartist", "TPE1"))
    album_raw = _first_value(tag_obj, ("album", "TALB"))
    title_raw = _first_value(tag_obj, ("title", "TIT2"))
    tn_raw = _first_value(tag_obj, ("tracknumber", "TRCK", "trkn"))

    return Tags(
        artist=_coerce_string(artist_raw),
        album=_coerce_string(album_raw),
        title=_coerce_string(title_raw),
        track_number=_parse_track_number(tn_raw),
    )
