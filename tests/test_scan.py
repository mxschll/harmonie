"""Tests for the recursive audio file walker."""

from __future__ import annotations

from pathlib import Path

from harmonie.scan import iter_audio_files, is_audio_file


def test_extension_detection(tmp_path: Path):
    a = tmp_path / "a.flac"
    a.write_text("x")
    b = tmp_path / "b.txt"
    b.write_text("x")
    assert is_audio_file(a)
    assert not is_audio_file(b)


def test_iter_recursive_and_dedupes(tmp_path: Path):
    (tmp_path / "sub" / "deep").mkdir(parents=True)
    f1 = tmp_path / "a.flac"
    f1.write_text("x")
    f2 = tmp_path / "sub" / "b.mp3"
    f2.write_text("x")
    f3 = tmp_path / "sub" / "deep" / "c.wav"
    f3.write_text("x")
    # Hidden directory should be ignored.
    (tmp_path / ".hidden").mkdir()
    skip = tmp_path / ".hidden" / "skip.flac"
    skip.write_text("x")

    found = sorted(p.name for p in iter_audio_files([tmp_path, tmp_path]))  # dup root
    assert found == ["a.flac", "b.mp3", "c.wav"]


def test_iter_handles_missing_root(tmp_path: Path):
    found = list(iter_audio_files([tmp_path / "does-not-exist"]))
    assert found == []


def test_split_library_path_finds_match(tmp_path: Path):
    from harmonie.scan import split_library_path

    lib = tmp_path / "music"
    (lib / "artist" / "album").mkdir(parents=True)
    track = lib / "artist" / "album" / "01.flac"
    track.write_text("x")

    root, rel = split_library_path(str(track), [lib])
    # split_library_path resolves both sides to absolute paths.
    assert root == str(lib.resolve())
    assert rel == "artist/album/01.flac"


def test_split_library_path_first_match_wins(tmp_path: Path):
    from harmonie.scan import split_library_path

    outer = tmp_path / "music"
    inner = outer / "sub"
    inner.mkdir(parents=True)
    track = inner / "x.flac"
    track.write_text("y")

    # First library that contains the track wins (outer).
    root, rel = split_library_path(str(track), [outer, inner])
    assert root == str(outer.resolve())
    assert rel == "sub/x.flac"


def test_split_library_path_returns_none_when_outside(tmp_path: Path):
    from harmonie.scan import split_library_path

    lib = tmp_path / "music"
    lib.mkdir()
    other = tmp_path / "elsewhere" / "track.flac"
    other.parent.mkdir()
    other.write_text("z")

    root, rel = split_library_path(str(other), [lib])
    assert root is None
    assert rel is None


def test_split_library_path_handles_trailing_slashes(tmp_path: Path):
    from harmonie.scan import split_library_path

    lib = tmp_path / "music"
    (lib / "a").mkdir(parents=True)
    track = lib / "a" / "t.flac"
    track.write_text("k")
    # Trailing slash on the configured library shouldn't trip the match.
    root, rel = split_library_path(str(track), [Path(str(lib) + "/")])
    assert root == str(lib.resolve())
    assert rel == "a/t.flac"
