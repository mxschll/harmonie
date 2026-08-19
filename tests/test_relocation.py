"""A library that moved must be matched to its existing rows, not re-analysed.

Analysis is keyed on absolute path, so remounting a library elsewhere — a native
install switching to a container, for instance — used to look like a brand new
library: every track analysed again, and the old rows left behind forever
because pruning only touches paths under the roots it scanned.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from harmonie import analyzer as analyzer_mod
from harmonie.analyzer import Analyzer
from harmonie.config import Settings
from harmonie.features import (
    DESCRIPTOR_VERSION,
    FINGERPRINT_CHUNK,
    MODEL_NAME,
    file_fingerprint,
)
from harmonie.workers import build_jobs


@pytest.fixture
def library(tmp_path: Path) -> Path:
    """A library with two tracks in a subdirectory."""
    root = tmp_path / "new-mount"
    (root / "Album").mkdir(parents=True)
    for i, name in enumerate(("01 - a.flac", "02 - b.flac")):
        # Distinct contents, so the two tracks have distinct fingerprints.
        (root / "Album" / name).write_bytes(f"audio-{i}-".encode() * 100)
    return root


def _index_as_if_scanned_at(
    db,
    old_root: Path,
    library: Path,
    rels,
    fake_descriptors,
    model: str = "m1",
    fingerprint: bool = False,
):
    """Insert rows as a scan under ``old_root`` would have written them, using
    the real files in ``library`` so size and mtime line up."""
    ids = []
    for rel in rels:
        stat = (library / rel).stat()
        ids.append(
            db.upsert_track(
                path=str(old_root / rel),
                library_root=str(old_root),
                relative_path=str(rel),
                fingerprint=(
                    file_fingerprint(library / rel, stat.st_size)
                    if fingerprint
                    else None
                ),
                size=stat.st_size,
                mtime=stat.st_mtime,
                duration=1.0,
                embedding=np.ones(4, dtype=np.float32),
                model=model,
                descriptors=fake_descriptors(),
                descriptor_version=DESCRIPTOR_VERSION,
            )
        )
    return ids


def test_fingerprint_survives_a_copy_that_loses_the_timestamp(tmp_path: Path):
    """The property the whole scheme rests on."""
    src = tmp_path / "track.flac"
    src.write_bytes(b"header" + b"x" * 300_000 + b"tail")
    copy = tmp_path / "elsewhere" / "track.flac"
    copy.parent.mkdir()
    copy.write_bytes(src.read_bytes())
    os.utime(copy, (0, 0))

    assert copy.stat().st_mtime != src.stat().st_mtime
    assert file_fingerprint(src) == file_fingerprint(copy)


def test_fingerprint_distinguishes_different_files(tmp_path: Path):
    a = tmp_path / "a.flac"
    b = tmp_path / "b.flac"
    a.write_bytes(b"A" + b"x" * 300_000)
    b.write_bytes(b"B" + b"x" * 300_000)
    assert file_fingerprint(a) != file_fingerprint(b)


def test_fingerprint_covers_the_tail_not_just_the_head(tmp_path: Path):
    """Files sharing a head and size must still be told apart — otherwise two
    tracks from one album rip could collide."""
    head = b"h" * (FINGERPRINT_CHUNK + 1000)
    a = tmp_path / "a.flac"
    b = tmp_path / "b.flac"
    a.write_bytes(head + b"a" * (FINGERPRINT_CHUNK + 1000))
    b.write_bytes(head + b"b" * (FINGERPRINT_CHUNK + 1000))
    assert a.stat().st_size == b.stat().st_size
    assert file_fingerprint(a) != file_fingerprint(b)


def test_fingerprint_handles_files_smaller_than_one_chunk(tmp_path: Path):
    small = tmp_path / "small.flac"
    small.write_bytes(b"tiny")
    empty = tmp_path / "empty.flac"
    empty.write_bytes(b"")

    assert file_fingerprint(small) != file_fingerprint(empty)
    assert file_fingerprint(small) == file_fingerprint(small)


def test_fingerprint_notices_a_size_change_alone(tmp_path: Path):
    """Size is hashed in, so truncation cannot pass as the same file."""
    f = tmp_path / "t.flac"
    f.write_bytes(b"x" * 1000)
    before = file_fingerprint(f)
    f.write_bytes(b"x" * 999)
    assert file_fingerprint(f) != before


def test_moved_library_is_matched_not_reanalysed(
    make_db, fake_descriptors, library: Path, tmp_path: Path
):
    db, _ = make_db()
    old_root = Path("/old/mount")
    tracks = sorted((library / "Album").iterdir())
    rel = [Path("Album") / t.name for t in tracks]
    _index_as_if_scanned_at(db, old_root, library, rel, fake_descriptors)
    assert db.get_track_by_path(str(old_root / "Album" / "01 - a.flac")) is not None

    settings = Settings(libraries=[library], data_dir=tmp_path)
    analyzer = Analyzer(settings)
    analyzer.db = db
    relocator = analyzer._build_relocator([library])
    assert relocator is not None, "a moved library should enable relocation"

    full, desc, skipped = build_jobs(
        db,
        tracks,
        model_name="m1",
        force=False,
        relocate=relocator,
    )

    assert relocator.moved == 2
    assert full == [], "nothing should need re-analysis"
    assert skipped == 2
    assert desc == []
    # The rows moved rather than multiplied.
    assert db.stats()["tracks"] == 2
    assert db.get_track_by_path(str(tracks[0])) is not None
    assert db.get_track_by_path(str(old_root / "Album" / "01 - a.flac")) is None


def test_unchanged_library_does_no_relocation_work(
    make_db, fake_descriptors, library: Path, tmp_path: Path
):
    """The cost for everyone who did not move anything: one DISTINCT query."""
    db, _ = make_db()
    tracks = sorted((library / "Album").iterdir())
    rel = [Path("Album") / t.name for t in tracks]
    _index_as_if_scanned_at(db, library, library, rel, fake_descriptors)

    analyzer = Analyzer(Settings(libraries=[library], data_dir=tmp_path))
    analyzer.db = db

    calls = {"relocation_rows": 0}
    original = db.relocation_rows

    def counting_rows(roots):
        calls["relocation_rows"] += 1
        return original(roots)

    db.relocation_rows = counting_rows

    assert analyzer._build_relocator([library]) is None
    assert calls["relocation_rows"] == 0, "no candidate rows should be read"

    # And the scan decision is untouched: everything already analysed.
    full, desc, skipped = build_jobs(
        db, tracks, model_name="m1", force=False, relocate=None
    )
    assert (full, desc, skipped) == ([], [], 2)


def test_changed_file_is_not_relocated(
    make_db, fake_descriptors, library: Path, tmp_path: Path
):
    """Same relative path but a different file must still be analysed."""
    db, _ = make_db()
    old_root = Path("/old/mount")
    tracks = sorted((library / "Album").iterdir())
    rel = [Path("Album") / t.name for t in tracks]
    ids = _index_as_if_scanned_at(db, old_root, library, rel, fake_descriptors)
    # Pretend the first file changed since it was indexed.
    stale = db.get_track_by_id(ids[0])["path"]
    db._conn.execute("UPDATE tracks SET size = 1 WHERE path = ?", (stale,))
    db._conn.commit()

    analyzer = Analyzer(Settings(libraries=[library], data_dir=tmp_path))
    analyzer.db = db
    relocator = analyzer._build_relocator([library])
    full, _desc, skipped = build_jobs(
        db, tracks, model_name="m1", force=False, relocate=relocator
    )

    assert relocator.moved == 1, "only the untouched file should be re-pointed"
    assert [j.path for j in full] == [str(tracks[0])]
    assert skipped == 1


def test_ambiguous_relative_paths_are_left_alone(
    make_db, fake_descriptors, library: Path, tmp_path: Path
):
    """Two old roots holding the same relative path: re-pointing either could
    attach an embedding to the wrong file."""
    db, _ = make_db()
    tracks = sorted((library / "Album").iterdir())
    rel = [Path("Album") / t.name for t in tracks]
    _index_as_if_scanned_at(db, Path("/old/one"), library, rel, fake_descriptors)
    _index_as_if_scanned_at(db, Path("/old/two"), library, rel, fake_descriptors)

    analyzer = Analyzer(Settings(libraries=[library], data_dir=tmp_path))
    analyzer.db = db
    relocator = analyzer._build_relocator([library])

    assert relocator is None, "ambiguous candidates should disable relocation"


def test_relocation_also_prevents_duplicates_under_force(
    make_db, fake_descriptors, library: Path, tmp_path: Path
):
    """A forced re-analysis should still reuse the existing row."""
    db, _ = make_db()
    old_root = Path("/old/mount")
    tracks = sorted((library / "Album").iterdir())
    rel = [Path("Album") / t.name for t in tracks]
    _index_as_if_scanned_at(db, old_root, library, rel, fake_descriptors)

    analyzer = Analyzer(Settings(libraries=[library], data_dir=tmp_path))
    analyzer.db = db
    relocator = analyzer._build_relocator([library])
    full, _desc, _skipped = build_jobs(
        db, tracks, model_name="m1", force=True, relocate=relocator
    )

    assert len(full) == 2, "force still re-analyses"
    assert relocator.moved == 2
    assert db.stats()["tracks"] == 2, "rows reused, not duplicated"


def test_relocation_declines_when_the_new_path_is_taken(
    make_db, fake_descriptors, library: Path, tmp_path: Path
):
    db, _ = make_db()
    tracks = sorted((library / "Album").iterdir())
    rel = [Path("Album") / t.name for t in tracks]
    _index_as_if_scanned_at(db, Path("/old/mount"), library, rel, fake_descriptors)
    # A row already lives at the new path.
    _index_as_if_scanned_at(db, library, library, rel[:1], fake_descriptors)

    analyzer = Analyzer(Settings(libraries=[library], data_dir=tmp_path))
    analyzer.db = db
    relocator = analyzer._build_relocator([library])
    build_jobs(db, tracks, model_name="m1", force=False, relocate=relocator)

    # The taken path was declined; the other file still relocated.
    assert relocator.moved == 1
    assert db.stats()["tracks"] == 3


def test_copied_library_with_rewritten_mtimes_is_matched_by_fingerprint(
    make_db, fake_descriptors, library: Path, tmp_path: Path
):
    """The case mtime cannot handle: copying a library to another system.

    `cp` without -p, a zip round trip, or a sync tool that ignores timestamps
    all leave the content intact and every mtime new.
    """
    db, _ = make_db()
    tracks = sorted((library / "Album").iterdir())
    rel = [Path("Album") / t.name for t in tracks]
    _index_as_if_scanned_at(
        db, Path("/old/mount"), library, rel, fake_descriptors, fingerprint=True
    )
    for t in tracks:
        os.utime(t, (0, 0))  # as a copy without -p would leave them

    analyzer = Analyzer(Settings(libraries=[library], data_dir=tmp_path))
    analyzer.db = db
    relocator = analyzer._build_relocator([library])
    full, desc, skipped = build_jobs(
        db, tracks, model_name="m1", force=False, relocate=relocator
    )

    assert relocator.matched_by_fingerprint == 2
    assert full == [], "content is unchanged, so nothing should be re-analysed"
    assert skipped == 2
    assert desc == []
    assert db.stats()["tracks"] == 2
    # mtime is refreshed, or the next check would send the file back again.
    row = db.get_track_by_path(str(tracks[0]))
    assert row["mtime"] == pytest.approx(tracks[0].stat().st_mtime)


def test_relative_path_match_reads_no_files(
    monkeypatch, make_db, fake_descriptors, library: Path, tmp_path: Path
):
    """The cheap key is tried first: a plain remount touches no file contents."""
    db, _ = make_db()
    tracks = sorted((library / "Album").iterdir())
    rel = [Path("Album") / t.name for t in tracks]
    _index_as_if_scanned_at(
        db, Path("/old/mount"), library, rel, fake_descriptors, fingerprint=True
    )

    analyzer = Analyzer(Settings(libraries=[library], data_dir=tmp_path))
    analyzer.db = db
    relocator = analyzer._build_relocator([library])

    reads = {"n": 0}
    real = analyzer_mod.file_fingerprint

    def counting(path, size=None):
        reads["n"] += 1
        return real(path, size)

    monkeypatch.setattr(analyzer_mod, "file_fingerprint", counting)
    build_jobs(db, tracks, model_name="m1", force=False, relocate=relocator)

    assert relocator.moved == 2
    assert reads["n"] == 0, "relative-path matches should not read file contents"


def test_fingerprints_are_backfilled_once(
    monkeypatch, make_db, fake_descriptors, library: Path, tmp_path: Path
):
    """Rows written before fingerprints existed get one, so a later copy of the
    library can be recognised. It must not happen on every scan."""
    db, _ = make_db()
    tracks = sorted((library / "Album").iterdir())
    rel = [Path("Album") / t.name for t in tracks]
    _index_as_if_scanned_at(db, library, library, rel, fake_descriptors)
    assert db.count_missing_fingerprints() == 2

    analyzer = Analyzer(Settings(libraries=[library], data_dir=tmp_path))
    analyzer.db = db

    reads = {"n": 0}
    real = analyzer_mod.file_fingerprint

    def counting(path, size=None):
        reads["n"] += 1
        return real(path, size)

    monkeypatch.setattr(analyzer_mod, "file_fingerprint", counting)

    assert analyzer._record_fingerprints(tracks) == 2
    assert reads["n"] == 2, "one read per un-fingerprinted track"
    assert db.count_missing_fingerprints() == 0

    # Second scan: nothing left to do, nothing read.
    assert analyzer._record_fingerprints(tracks) == 0
    assert reads["n"] == 2


def test_ambiguous_fingerprints_are_left_alone(
    make_db, fake_descriptors, library: Path, tmp_path: Path
):
    """Two rows with one fingerprint cannot identify a file."""
    db, _ = make_db()
    tracks = sorted((library / "Album").iterdir())
    # Both files hold identical bytes, so they share a fingerprint.
    tracks[1].write_bytes(tracks[0].read_bytes())
    rel = [Path("Album") / t.name for t in tracks]
    _index_as_if_scanned_at(
        db, Path("/old/mount"), library, rel, fake_descriptors, fingerprint=True
    )
    for t in tracks:
        os.utime(t, (0, 0))

    analyzer = Analyzer(Settings(libraries=[library], data_dir=tmp_path))
    analyzer.db = db
    relocator = analyzer._build_relocator([library])
    full, _desc, _skipped = build_jobs(
        db, tracks, model_name="m1", force=False, relocate=relocator
    )

    assert relocator.matched_by_fingerprint == 0
    assert len(full) == 2, "ambiguous rows must not be re-pointed"


def test_relocator_is_offered_during_a_real_scan(
    monkeypatch, make_db, fake_descriptors, library: Path, tmp_path: Path
):
    """The wiring, not just the helper: _run_scan must pass the relocator."""
    db, _ = make_db()
    tracks = sorted((library / "Album").iterdir())
    rel = [Path("Album") / t.name for t in tracks]
    _index_as_if_scanned_at(
        db, Path("/old/mount"), library, rel, fake_descriptors, model=MODEL_NAME
    )

    analyzer = Analyzer(Settings(libraries=[library], data_dir=tmp_path))
    analyzer.db = db

    seen = {}
    original = analyzer_mod.build_jobs

    def spy(db_, files, **kwargs):
        seen["relocate"] = kwargs.get("relocate")
        return original(db_, files, **kwargs)

    monkeypatch.setattr(analyzer_mod, "build_jobs", spy)
    analyzer.scan()

    assert isinstance(seen["relocate"], analyzer_mod._LibraryRelocator)
    assert seen["relocate"].moved == 2
    assert analyzer.status.skipped == 2
    assert db.stats()["tracks"] == 2
