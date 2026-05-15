"""Tests for tag extraction. Uses mocked mutagen at the unit level — the
real-file integration is exercised by the smoke test in CI / by hand."""

from __future__ import annotations

from pathlib import Path

import pytest

from harmonie.tags import (
    Tags,
    _coerce_string,
    _first_value,
    _parse_track_number,
    extract_tags,
)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_parse_track_number_variants():
    assert _parse_track_number("5") == 5
    assert _parse_track_number("5/12") == 5
    assert _parse_track_number(["7"]) == 7
    assert _parse_track_number((3, 12)) == 3
    assert _parse_track_number(["7/12"]) == 7
    assert _parse_track_number(None) is None
    assert _parse_track_number("") is None
    assert _parse_track_number("notanumber") is None
    assert _parse_track_number("  4  ") == 4


def test_coerce_string():
    assert _coerce_string("hello") == "hello"
    assert _coerce_string(["alpha", "beta"]) == "alpha"
    assert _coerce_string(b"bytes-text") == "bytes-text"
    assert _coerce_string([]) is None
    assert _coerce_string(None) is None
    assert _coerce_string("   ") is None


def test_first_value_iteration():
    d = {"artist": ["Person"], "albumartist": ["Other"]}
    assert _first_value(d, ["artist", "albumartist"]) == ["Person"]
    assert _first_value(d, ["missing", "albumartist"]) == ["Other"]
    assert _first_value(d, ["x", "y"]) is None
    assert _first_value(None, ["any"]) is None


# ---------------------------------------------------------------------------
# extract_tags with mocked mutagen
# ---------------------------------------------------------------------------


class _FakeFile:
    def __init__(self, tags):
        self.tags = tags


def test_extract_tags_vorbis_style(monkeypatch):
    """Vorbis comments / EasyID3 / EasyMP4 all expose dict-like keys."""
    fake = _FakeFile(
        {
            "artist": ["Aphex Twin"],
            "album": ["Selected Ambient Works"],
            "title": ["Xtal"],
            "tracknumber": ["1/13"],
            "musicbrainz_trackid": ["e3a06aa1-22f1-4d33-9eb9-5a13b48f12f1"],
        }
    )
    import mutagen

    monkeypatch.setattr(mutagen, "File", lambda p, easy=False: fake)

    t = extract_tags(Path("/whatever.flac"))
    assert t == Tags(
        artist="Aphex Twin",
        album="Selected Ambient Works",
        title="Xtal",
        track_number=1,
        musicbrainz_track_id="e3a06aa1-22f1-4d33-9eb9-5a13b48f12f1",
    )


def test_extract_tags_falls_back_to_albumartist(monkeypatch):
    fake = _FakeFile({"albumartist": ["Various"], "title": ["Untitled"]})
    import mutagen

    monkeypatch.setattr(mutagen, "File", lambda p, easy=False: fake)
    t = extract_tags(Path("/x.mp3"))
    assert t.artist == "Various"
    assert t.title == "Untitled"
    assert t.album is None


def test_extract_tags_returns_empty_when_mutagen_returns_none(monkeypatch):
    import mutagen

    monkeypatch.setattr(mutagen, "File", lambda p, easy=False: None)
    assert extract_tags(Path("/wat")) == Tags()


def test_extract_tags_returns_empty_on_mutagen_error(monkeypatch):
    import mutagen

    def boom(*a, **k):
        raise RuntimeError("mutagen exploded")

    monkeypatch.setattr(mutagen, "File", boom)
    assert extract_tags(Path("/wat")) == Tags()


def test_extract_tags_no_tags_attribute(monkeypatch):
    """Some formats parse but expose no .tags (e.g. tagless WAV)."""

    class TaglessFile:
        tags = None

    import mutagen

    monkeypatch.setattr(mutagen, "File", lambda p, easy=False: TaglessFile())
    assert extract_tags(Path("/x.wav")) == Tags()
