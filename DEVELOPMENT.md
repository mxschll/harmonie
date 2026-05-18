# Development

Architecture, scaling notes, and contribution workflow for harmonie. See [README.md](README.md) for installation and API usage.

## Contents

- [Local setup](#local-setup)
- [How it works](#how-it-works)
- [Architecture](#architecture)
- [Scaling](#scaling)
- [Lint and format](#lint-and-format)
- [Tests](#tests)
- [Scan history](#scan-history)
- [Cancellation](#cancellation)
- [Schema migrations](#schema-migrations)
- [Releases](#releases)

## Local setup

```bash
git clone https://github.com/mxschll/harmonie.git
cd harmonie
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --pre -e ".[dev]"

export HARMONIE_LIBRARIES="/path/to/music"
export HARMONIE_DATA_DIR="./data"
harmonie serve
```

CLI for one-shot operations against the same data dir:

```bash
harmonie scan
harmonie status
harmonie list --bpm-min 120 --bpm-max 130
harmonie similar 1 -n 10
harmonie migrate
```

## How it works

1. **Scan.** `harmonie.scan` walks the configured roots and yields audio files (FLAC, MP3, WAV, OGG, M4A, AAC, AIFF, OPUS, WMA, ALAC).
2. **Schedule.** A coroutine triggers `analyzer.scan()` on startup and every `HARMONIE_SCAN_INTERVAL_HOURS`. A second `POST /api/v1/scan` while one is running is a no-op.
3. **Workers.** Files go through a `multiprocessing.Pool` of N processes. Each worker loads the model once at startup and reuses it. Two job types: full extraction (embedding + descriptors) and descriptor-only (top-up an existing row when only the descriptor pipeline changed).
4. **Extract.** Each file is decoded once at 44.1 kHz mono. `RhythmExtractor2013`, `KeyExtractor`, `ReplayGain`, `Danceability`, and `OnsetRate` give the descriptor block. The same audio is resampled in memory to 16 kHz for `TensorflowPredictEffnetDiscogs`, whose 1280-d penultimate-layer outputs are averaged across windows. Tags (artist, album, title, track number) are read in parallel via `mutagen`.
5. **Store.** SQLite (WAL mode) keeps one row per track: path, `library_root` + `relative_path`, size+mtime for change detection, embedding blob, descriptor columns, tag columns, `model` and `descriptor_version` for cheap top-ups. Filter columns are indexed.
6. **Search.** Similarity queries hit an in-memory L2-normalised matrix kept by `EmbeddingIndex` (rebuilt lazily after each scan). A query is a single matrix-vector multiply; optional descriptor filters gate candidates before ranking.
7. **Prune.** Files that disappeared between scans are dropped from the DB. The prune is scoped to roots that were reachable in this scan, so a temporarily-offline mount won't wipe the index.

## Architecture

The data flow is one-way and the layers are thin.

```
filesystem  →  scanner  →  worker pool  →  Database  →  EmbeddingIndex  →  HTTP
                                          (SQLite)    (in-memory cache)    (FastAPI)
```

* `Database` is the single source of truth: one SQLite file, one row per track, embedding stored as a `float32` blob plus indexed descriptor columns. Reads run concurrent with writes thanks to WAL mode.
* `EmbeddingIndex` is a process-wide cache keyed by model. It stores L2-normalised matrices in RAM so similarity queries are pure matmuls (~30 µs on small libraries vs ~2 ms reading from SQLite). The cache is invalidated wholesale at the end of every scan; the next query rebuilds lazily.
* `Analyzer` owns both for the service lifetime. The HTTP layer never opens its own DB. Handlers depend on the analyzer's instances via FastAPI dependency injection.
* `similarity.py` and `playlist.py` are stateless query layers. They take a `Database` (for descriptor metadata) and an `EmbeddingIndex` (for vectors) and return result objects. Swapping the index for a FAISS-backed implementation later is a single-file change.

The two-version split (`model` vs `descriptor_version`) lets a descriptor-algorithm bump re-analyse just the descriptor columns without re-running TensorFlow on every track. The worker pool honours that distinction with two job types.

## Scaling

* Brute-force similarity holds up to ~250k tracks. Above that, swap `harmonie.similarity` for FAISS or hnswlib. The API is small.
* 100k tracks × 1280 floats = 512 MB of embeddings in memory plus the model and overhead. Expect 1.5–2 GB RSS.
* Initial scans of large libraries are CPU-bound at ~4 s/track. Scale `HARMONIE_WORKERS` to your core count; each worker holds its own copy of the model (~200 MB).
* Per-worker memory peaks well above the 200 MB baseline during extraction. A 30-minute FLAC decoded to 44.1 kHz mono float32 is ~317 MB on its own; resampling and TF inference push the working set close to 1 GB per worker on long files. On RAM-constrained boxes (or any system with no swap, where systemd-oomd is aggressive about memory pressure), set `HARMONIE_WORKERS` low enough that `workers × 1 GB` fits comfortably in available RAM. 6 workers on a 24 GB box is a safe starting point for libraries with classical recordings.
* SQLite is fine to several million rows. Move to Postgres only if you want multiple service instances sharing one DB.

## Lint and format

Ruff handles both. Configuration lives in `[tool.ruff]` in `pyproject.toml`.

```bash
ruff check .          # lint
ruff format .         # apply formatting
ruff format --check . # CI-style: fail if anything would change
ruff check --fix .    # auto-fix what's safely fixable
```

CI runs `ruff check` and `ruff format --check` on every push and pull request via `.github/workflows/lint.yml`. Failures block the build.

## Tests

```bash
pip install --pre -e ".[dev]"
pytest
```

The non-Essentia parts (DB, similarity, playlist, scan, tags, API) are covered by the `tests/` suite. Essentia itself is exercised by running an actual scan against a small fixture library.

## Scan history

Every scan run is persisted to two tables (added in migration 002):

* `scans` — one row per scan: `started_at`, `finished_at`, `duration_sec`, the counters (discovered/full/descriptors_only/skipped/failed/removed), the configuration at the time (`workers`, `model`, `forced`, `harmonie_version`, `descriptor_version`), and an outcome (`state` is `running` | `completed` | `crashed` | `cancelled`, plus `last_error` on hard failures). The `backend` column is kept for historical compat but always reads `effnet`.
* `scan_failures` — one row per failed track per scan, linked to `scans.id` with `ON DELETE CASCADE`.

If the process is killed mid-scan, the `scans` row is left in `state='running'`. The next `Analyzer()` instance marks any such rows as `crashed` on construction with a synthetic `finished_at` and a `last_error = 'interrupted before completion'`.

The `harmonie scans` CLI command is the debug surface:

```bash
harmonie scans                # list recent scans (default 10)
harmonie scans --limit 50     # list more
harmonie scans 14             # details + every failed track for scan 14
harmonie scans 14 --json      # same, machine-readable
```

There is no API endpoint for scan history. Use the CLI on the host, or read the SQLite file directly.

## Cancellation

A scan can be aborted at any time. The mechanism is the same in three contexts:

* **`harmonie scan` CLI** — Ctrl-C once cancels gracefully (workers terminated, partial results committed, scan row marked `cancelled`). Ctrl-C twice force-exits via `os._exit(130)`.
* **`harmonie serve` graceful shutdown** — uvicorn's SIGINT handling triggers our FastAPI lifespan teardown, which calls `analyzer.request_cancel()` before `analyzer.stop()`. The pool is terminated rather than drained, so an in-flight scan with a long queue (50k+ jobs) doesn't block shutdown.
* **Programmatically** — `analyzer.request_cancel()` sets a `threading.Event`, terminates the worker pool, and returns. The result loop in `_run_scan` checks the event each iteration and breaks out cleanly.

Cancellation persists to the `scans` table as `state = 'cancelled'`, distinct from `crashed` (which means an unhandled exception). The `prune` phase is skipped on cancel because the file enumeration may be incomplete and pruning could otherwise drop rows for files that just weren't enumerated yet.

The pool is terminated (`SIGTERM` to all workers) rather than closed-and-joined. Workers stuck on slow I/O — common with network mounts like CIFS or NFS — get killed instead of waited on. Any tracks in the middle of extraction at cancel time are simply lost; the next scan picks them up via the standard size+mtime incremental check.

## Schema migrations

The schema lives in the `harmonie/migrations/` package, with one file per version:

```
harmonie/migrations/
├── __init__.py             # discovery + runner + public API
├── m001_initial.py         # def upgrade(conn): ...
├── m002_scan_history.py    # def upgrade(conn): ...
```

On import, the package scans itself for `mNNN_*.py` modules, sorts them by `NNN`, validates that the versions form a contiguous sequence starting at 1, and exposes the resulting list as `MIGRATIONS`. The runner is invoked automatically when the DB is opened (so `harmonie scan`, `harmonie serve`, and any test that constructs a `Database` always sees an up-to-date schema), and is also exposed as a one-shot CLI:

```bash
harmonie migrate
# Migrated database from version 0 to 2.
harmonie migrate
# Already at schema version 2. Nothing to do.
```

Each migration runs in its own transaction. A failure mid-migration rolls back to the previous version and leaves the DB usable. The runner refuses to operate on a DB that has been migrated past the latest version known to this binary; that would risk silent data loss.

To add a migration:

1. Create `harmonie/migrations/mNNN_short_description.py` where `NNN` is the next version number, zero-padded to three digits.
2. Define `def upgrade(conn: sqlite3.Connection) -> None:` and put the DDL inside (typically as a list of statements run in a loop).
3. Update any code in `db.py` that depends on the new shape (column lists in `upsert_track`, `list_tracks`, `get_tracks_by_ids`, etc.).

Existing migration files don't change after they've shipped; new changes are new migrations. Migration `upgrade` functions must use `conn.execute(...)` for each statement individually rather than `conn.executescript(...)`. The latter issues its own commit and breaks the surrounding transaction.


## Releases

Versions come from git tags via setuptools-scm. A push to `main` automatically creates a new patch tag (`v1.0.1`, `v1.0.2`, ...) and a corresponding GitHub release with auto-generated notes. The workflow lives in `.github/workflows/release.yml`.

The auto-bump only ever steps the patch component. For minor or major bumps, tag the desired commit yourself before pushing:

```bash
# After merging a feature:
git tag v1.1.0
git push origin v1.1.0

# After a breaking change:
git tag v2.0.0
git push origin v2.0.0
```

The next push to `main` after that resumes patch-bumping from your tag (`v1.1.0` → `v1.1.1`, etc.).

To skip versioning for a single commit (typo fixes, CI tweaks, doc-only changes), include `[skip release]` in the commit message:

```bash
git commit -m "docs: fix typo [skip release]"
```

`[skip ci]` works as a synonym.

`pipx upgrade harmonie` always picks up the highest version tag, so users see the new release the moment the tag is pushed.
