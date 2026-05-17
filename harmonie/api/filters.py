"""Filter language for the HTTP API.

Two surfaces map to a single internal :class:`harmonie.db.TrackFilter`:

* **URL queries** (used by ``GET /tracks`` and ``GET /tracks/{id}/similar``):

  * Numeric ranges via ``..`` syntax::

      bpm=120..130     # closed range
      bpm=120..        # lower bound only
      bpm=..130        # upper bound only
      bpm=128          # exact value

  * Set membership: repeat the parameter (``key=A&key=B``).

  * Style filter is the same: ``style=Electronic`` (prefix on the genre side)
    or ``style=Electronic---House`` (exact). ``style_min`` gates by minimum
    classifier probability; ``style_mode=any|all`` switches between OR and
    AND semantics.

* **JSON bodies** (used by ``POST /playlists`` under ``filter``)::

      {
        "bpm":      { "gte": 120, "lte": 130 },
        "loudness": { "lte": -10 },
        "key":      ["A", "B"],
        "scale":    "minor",
        "style":    ["Electronic"],
        "style_min": 0.5,
        "style_mode": "any"
      }

Both shapes build the same ``TrackFilter``.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, model_validator

from ..db import TrackFilter


# ---------------------------------------------------------------------------
# Range objects (used in body filters)
# ---------------------------------------------------------------------------


class FloatRange(BaseModel):
    """Inclusive numeric range. Either bound may be omitted."""

    gte: Optional[float] = None
    lte: Optional[float] = None

    @model_validator(mode="after")
    def _check_bounds(self) -> "FloatRange":
        if self.gte is not None and self.lte is not None and self.gte > self.lte:
            raise ValueError(f"gte ({self.gte}) must be <= lte ({self.lte})")
        return self

    def is_empty(self) -> bool:
        return self.gte is None and self.lte is None


# ---------------------------------------------------------------------------
# Body filter
# ---------------------------------------------------------------------------


class FilterBody(BaseModel):
    """Body shape for ``filter`` blocks in playlist requests.

    Every field is optional. Missing fields mean "no constraint."
    """

    bpm: Optional[FloatRange] = None
    danceability: Optional[FloatRange] = None
    loudness: Optional[FloatRange] = None
    key: Optional[list[str]] = None
    scale: Optional[str] = None
    style: Optional[list[str]] = Field(
        None,
        description=(
            "Discogs-400 style filter. Each entry is either a full "
            "``Genre---Style`` label (exact) or a bare ``Genre`` (prefix on "
            "the whole branch)."
        ),
    )
    style_min: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Minimum classifier probability for a style row to count.",
    )
    style_mode: str = Field(
        "any", pattern="^(any|all)$",
        description="``any`` (default) or ``all`` of the requested styles.",
    )

    def to_track_filter(self) -> TrackFilter:
        bpm = self.bpm or FloatRange()
        dance = self.danceability or FloatRange()
        loud = self.loudness or FloatRange()
        return TrackFilter(
            bpm_min=bpm.gte,
            bpm_max=bpm.lte,
            danceability_min=dance.gte,
            danceability_max=dance.lte,
            loudness_min=loud.gte,
            loudness_max=loud.lte,
            key=self.key,
            scale=self.scale,
            styles=self.style,
            style_min_probability=self.style_min,
            style_match=self.style_mode,
        )


# ---------------------------------------------------------------------------
# URL range parser
# ---------------------------------------------------------------------------


def parse_range(value: Optional[str]) -> FloatRange:
    """Parse a query-string range into a :class:`FloatRange`.

    Accepts ``"120..130"`` (closed), ``"120.."`` (lower only), ``"..130"``
    (upper only), or a bare number ``"128"`` (treated as ``gte=lte=128``).
    Returns an empty range when ``value`` is ``None`` or empty.

    Raises ``ValueError`` on a malformed input — caller maps to HTTP 400.
    """
    if value is None or value == "":
        return FloatRange()
    if ".." in value:
        lo_str, hi_str = value.split("..", 1)
        lo = float(lo_str) if lo_str else None
        hi = float(hi_str) if hi_str else None
        return FloatRange(gte=lo, lte=hi)
    # Bare number = exact match (closed degenerate range).
    n = float(value)
    return FloatRange(gte=n, lte=n)


def build_track_filter(
    *,
    bpm: Optional[str] = None,
    danceability: Optional[str] = None,
    loudness: Optional[str] = None,
    key: Optional[list[str]] = None,
    scale: Optional[str] = None,
    style: Optional[list[str]] = None,
    style_min: float = 0.0,
    style_mode: str = "any",
) -> TrackFilter:
    """Compose a :class:`TrackFilter` from URL-style query parameters.

    Numeric range params accept the ``120..130`` / ``120..`` / ``..130`` /
    ``128`` shorthand documented in :func:`parse_range`. Set-membership params
    (``key``, ``style``) are passed through as lists.
    """
    bpm_r = parse_range(bpm)
    dance_r = parse_range(danceability)
    loud_r = parse_range(loudness)
    return TrackFilter(
        bpm_min=bpm_r.gte,
        bpm_max=bpm_r.lte,
        danceability_min=dance_r.gte,
        danceability_max=dance_r.lte,
        loudness_min=loud_r.gte,
        loudness_max=loud_r.lte,
        key=key,
        scale=scale,
        styles=style,
        style_min_probability=style_min,
        style_match=style_mode,
    )
