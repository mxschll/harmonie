# Development

Architecture, scaling notes, and contribution workflow for harmonie. See [README.md](README.md) for installation and API usage.

## Contents

- [Local setup](#local-setup)
- [How it works](#how-it-works)
- [Architecture](#architecture)
- [Scaling](#scaling)
- [Lint and format](#lint-and-format)
- [Tests](#tests)
- [Schema migrations](#schema-migrations)

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

## Schema migrations

The schema lives in `harmonie/migrations.py` as an append-only list of versioned migration functions. The runner is invoked automatically when the DB is opened (so `harmonie scan`, `harmonie serve`, and any test that constructs a `Database` always sees an up-to-date schema), and is also exposed as a one-shot CLI:

```bash
harmonie migrate
# Migrated database from version 0 to 1.
harmonie migrate
# Already at schema version 1. Nothing to do.
```

Each migration runs in its own transaction. A failure mid-migration rolls back to the previous version and leaves the DB usable. The runner refuses to operate on a DB that has been migrated past the latest version known to this binary; that would risk silent data loss.

To add a migration:

1. Add a `_migration_NNN_what_it_does(conn)` function to `harmonie/migrations.py`.
2. Append it to the `MIGRATIONS` list in the same file.
3. Update any code in `db.py` that depends on the new shape (column lists in `upsert_track`, `list_tracks`, `get_tracks_by_ids`, etc.).

Migration functions must use `conn.execute(...)` for each statement individually rather than `conn.executescript(...)`. The latter issues its own commit and breaks the surrounding transaction.
