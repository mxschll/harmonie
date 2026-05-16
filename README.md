# harmonie

[![Tests](https://github.com/mxschll/harmonie/actions/workflows/tests.yml/badge.svg)](https://github.com/mxschll/harmonie/actions/workflows/tests.yml)
[![Docker](https://github.com/mxschll/harmonie/actions/workflows/docker.yml/badge.svg)](https://github.com/mxschll/harmonie/actions/workflows/docker.yml)

Audio similarity service. Scans a music library, extracts a per-track embedding plus musical descriptors (BPM, key, loudness, danceability, onset rate) and reads the file's tags (artist, album, title, track number) using [Essentia](https://essentia.upf.edu/) and [mutagen](https://mutagen.readthedocs.io/), stores everything in SQLite, and exposes an HTTP API for similarity queries and playlist generation.

The default backend is Essentia's **Discogs-Effnet** model — a 1280-d embedding trained on Discogs tags that's well suited to music similarity. A lighter `MusicExtractor` backend is available for hosts without TensorFlow.

The intended deployment is a long-running container that periodically rescans the library and serves any HTTP client — a media-server plugin, a custom playlist generator, a CLI tool — that wants similarity queries against the indexed catalog.

## Quick start (Docker)

A `linux/amd64` image is published to GitHub Container Registry on every push to `main` and on each version tag:

```bash
docker pull ghcr.io/mxschll/harmonie:latest
```

> The image is amd64-only because Essentia ships only manylinux x86_64 wheels on PyPI. Apple Silicon and other arm64 hosts run the image transparently via Docker's emulation layer; Pi-class arm64 hosts that can't tolerate emulation can install directly with `pip` (see the local-Python instructions below).

Available tags:

| Tag                    | What it tracks                                |
| ---------------------- | --------------------------------------------- |
| `latest`               | Latest commit on `main`                       |
| `main`                 | Same as `latest`                              |
| `0.1.<n>`              | Auto-versioned per push to `main` (`n` = workflow run number) |
| `sha-<short>`          | A specific commit                             |
| `vX.Y.Z` / `X.Y` / `X` | Manually-tagged release (`git tag vX.Y.Z`)    |

To run with `docker compose`, copy the example config and point it at your library:

```bash
cp .env.example .env
# Edit .env: set HARMONIE_LIBRARIES (path inside the container) and edit
# docker-compose.yml to mount your library at that path.

docker compose up -d
docker compose logs -f harmonie
```

The service stores its DB in `./data/harmonie.db`. On first start it downloads the Discogs-Effnet model (~18 MB) into `~/.cache/harmonie/models/` inside the container.

## Quick start (local Python)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"

# Point at a library, then run the service.
export HARMONIE_LIBRARIES="/path/to/music"
export HARMONIE_DATA_DIR="./data"
harmonie serve
```

Or run a one-shot scan from the CLI:

```bash
harmonie scan
harmonie status
harmonie list --bpm-min 120 --bpm-max 130
harmonie similar 1 -n 10
```

## API

All endpoints are versioned under `/api/v1/`. If `HARMONIE_API_KEY` is set, every authenticated request must include `X-API-Key: <key>`. `GET /healthz` is always public.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET`  | `/healthz` | Liveness probe |
| `GET`  | `/api/v1/status` | Service version, library stats, last scan |
| `POST` | `/api/v1/scan` | Trigger a scan now (`{"force": false}`) |
| `GET`  | `/api/v1/scan/status` | Scan progress and counters |
| `GET`  | `/api/v1/tracks` | List tracks (filters + pagination) |
| `GET`  | `/api/v1/tracks/{id}` | Full track record |
| `GET`  | `/api/v1/tracks/{id}/similar` | Top-N similar tracks |
| `POST` | `/api/v1/tracks/lookup` | Find a single track by `path` and/or tags |
| `POST` | `/api/v1/playlists` | Build a playlist (mode implicit from parameters) |

OpenAPI docs are served at `/docs` (Swagger UI) and `/openapi.json`.

### Mapping harmonie tracks to an external catalog

Every track and every match in API responses includes the metadata you need to look it up in another system without doing a filesystem walk.

* **`artist` + `album` + `title` + `track_number`** — the tag-based match. The four fields together are usually enough to identify a track unambiguously in another catalog.
* **`library_root` + `relative_path`** — the path-based match. If the consumer sees the same library layout under a different mount point, it joins on `relative_path` directly. No path-prefix mapping config needed in the common case.

`library_root` reflects the configured `HARMONIE_LIBRARIES` entries at scan time. If you reconfigure mount points, re-scan to refresh.

The `POST /api/v1/tracks/lookup` endpoint exposes this matching directly. Send any subset of `{path, artist, album, title}` and get back a single track:

```bash
# By tags (case-insensitive).
curl -X POST http://localhost:8842/api/v1/tracks/lookup \
  -H 'content-type: application/json' \
  -d '{"artist": "Aphex Twin", "album": "SAW", "title": "Xtal"}'

# By path (works against the absolute path or the relative_path).
curl -X POST http://localhost:8842/api/v1/tracks/lookup \
  -H 'content-type: application/json' \
  -d '{"path": "Aphex Twin/SAW/01 Xtal.flac"}'
```

The lookup tries strategies in order — exact path, relative path, full tag triple, looser tag pair — and returns the first match (smallest id wins on ties). 404 if nothing matches; 400 if you send an empty body.

### Filter parameters

Both `/tracks` and `/tracks/{id}/similar` accept the same set of optional descriptor filters as query params:

```
bpm_min, bpm_max, key (repeatable), scale,
danceability_min, danceability_max, loudness_min, loudness_max
```

For playlist endpoints the same set is in the body under `filter`.

### Playlists

One endpoint, `POST /api/v1/playlists`, builds every kind of playlist. The mode is implicit from the parameters you set:

* **No `seeds`** → descriptor-driven. The candidate pool is whatever passes the `filter`; results are sorted by closeness to `target_bpm` / `target_danceability` and optionally shuffled.
* **`seeds` set, `drift: false`** (default) → similarity-driven. The seeds anchor the playlist; results stay near them in embedding space. `bpm_tolerance` and `key_compatible` add smooth-transition rules.
* **One seed, `drift: true`** → drifting walk. Take the top `chunk_size` tracks similar to the seed, then re-anchor on the *last* of those and take its top `chunk_size`, and so on until the playlist hits `n`. Larger `chunk_size` stays closer to the seed; smaller `chunk_size` drifts faster. No track ever appears twice.

Parameters:

| Field | Type | Purpose |
| --- | --- | --- |
| `n` | int (1–500) | How many tracks to return. Default 20. |
| `seeds` | list[int] | Track IDs to anchor on. Empty = descriptor-driven. |
| `drift` | bool | Walk away from the seed instead of staying near it. Requires exactly one seed. |
| `chunk_size` | int (1–100) | Tracks per anchor in drift mode. Default 5. |
| `filter` | object | Hard descriptor constraints — same shape as `/tracks` query params. |
| `bpm_tolerance` | float | Max BPM gap between consecutive tracks. Seeds-only. |
| `key_compatible` | bool | Restrict to keys that mix harmonically with the first seed (Camelot wheel: same key, ±1 number, parallel mode). Seeds-only. |
| `target_bpm` | float | Pull tracks toward this BPM when ranking. No-seeds-only. |
| `target_danceability` | float | Pull tracks toward this danceability score when ranking. No-seeds-only. |
| `include_seeds` | bool | Include seed tracks in the result. |
| `shuffle` | bool | Randomise order. No-seeds-only. Default true. |
| `rng_seed` | int | Reproducible shuffle. |

Examples:

```bash
# Similar to track 1, max ±5 BPM jumps, harmonically compatible keys.
curl -X POST http://localhost:8842/api/v1/playlists \
  -H 'content-type: application/json' \
  -d '{
    "seeds": [1],
    "n": 20,
    "bpm_tolerance": 5,
    "key_compatible": true
  }'

# Drift away from track 1 in chunks of 3, total length 25.
curl -X POST http://localhost:8842/api/v1/playlists \
  -H 'content-type: application/json' \
  -d '{"seeds": [1], "drift": true, "chunk_size": 3, "n": 25}'

# Descriptor-driven: 30 tracks at ~128 BPM, danceability ≥ 1.5.
curl -X POST http://localhost:8842/api/v1/playlists \
  -H 'content-type: application/json' \
  -d '{
    "n": 30,
    "target_bpm": 128,
    "filter": {"danceability_min": 1.5}
  }'
```

## Configuration

All settings come from environment variables (or a `.env` file in the working directory):

| Variable | Default | Purpose |
| --- | --- | --- |
| `HARMONIE_LIBRARIES` | (none) | Comma- or colon-separated absolute paths to scan |
| `HARMONIE_DATA_DIR` | `./data` | Where to put `harmonie.db` |
| `HARMONIE_BACKEND` | `effnet` | `effnet` or `musicextractor` |
| `HARMONIE_WORKERS` | CPU count | Analysis worker processes |
| `HARMONIE_SCAN_INTERVAL_HOURS` | `6` | Periodic scan interval (`0` disables) |
| `HARMONIE_SCAN_ON_STARTUP` | `true` | Run a scan immediately on boot |
| `HARMONIE_HOST` | `0.0.0.0` | HTTP bind address |
| `HARMONIE_PORT` | `8842` | HTTP port |
| `HARMONIE_API_KEY` | (none) | If set, required in `X-API-Key` |
| `HARMONIE_CORS_ORIGINS` | (none) | Comma-separated list to enable CORS |
| `HARMONIE_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `HARMONIE_LOG_JSON` | `false` | Emit one-line JSON logs |

## How it works

1. **Scan.** `harmonie.scan` walks the configured roots and yields audio files (FLAC, MP3, WAV, OGG, M4A, AAC, AIFF, OPUS, WMA, ALAC).
2. **Schedule.** A coroutine triggers `analyzer.scan()` on startup and every `HARMONIE_SCAN_INTERVAL_HOURS`. A second `POST /api/v1/scan` while one is running is a no-op.
3. **Workers.** Files go through a `multiprocessing.Pool` of N processes. Each worker loads the model once at startup and reuses it. Two job types: full extraction (embedding + descriptors) and descriptor-only (top-up an existing row when only the descriptor pipeline changed).
4. **Extract.** Each file is decoded once at 44.1 kHz mono. `RhythmExtractor2013`, `KeyExtractor`, `ReplayGain`, `Danceability`, and `OnsetRate` give the descriptor block. The same audio is resampled in memory to 16 kHz for `TensorflowPredictEffnetDiscogs`, whose 1280-d penultimate-layer outputs are averaged across windows. **Tags** (artist, album, title, track number) are read in parallel via `mutagen`.
5. **Store.** SQLite (WAL mode) keeps one row per track: path, `library_root` + `relative_path`, size+mtime for change detection, embedding blob, descriptor columns, tag columns, `model` and `descriptor_version` for cheap top-ups. Filter columns are indexed.
6. **Search.** Similarity queries hit an in-memory L2-normalised matrix kept by `EmbeddingIndex` (rebuilt lazily after each scan). A query is a single matrix-vector multiply; optional descriptor filters gate candidates before ranking.
7. **Prune.** Files that disappeared between scans are dropped from the DB. The prune is scoped to roots that were actually reachable, so a temporarily-offline mount won't wipe the index.

## Architecture

The data flow is one-way and the layers are intentionally thin.

```
filesystem  →  scanner  →  worker pool  →  Database  →  EmbeddingIndex  →  HTTP
                                          (SQLite)    (in-memory cache)    (FastAPI)
```

* **`Database`** is the single source of truth: one SQLite file, one row per track, embedding stored as a `float32` blob plus indexed descriptor columns. Reads are concurrent with writes thanks to WAL mode.
* **`EmbeddingIndex`** is a process-wide cache, keyed by model. It stores L2-normalised matrices in RAM so similarity queries are pure matmuls (~30 µs on small libraries vs. ~2 ms reading from SQLite). The cache is invalidated wholesale at the end of every scan; the next query rebuilds lazily.
* **`Analyzer`** owns both for the service lifetime. The HTTP layer never opens its own DB — handlers depend on the analyzer's instances via FastAPI dependency injection.
* **`similarity.py`** and **`playlist.py`** are stateless query layers. They take a `Database` (for descriptor metadata) and an `EmbeddingIndex` (for vectors) and return result objects. Swapping the index for a FAISS-backed implementation later is a single-file change.

The two-version split (`model` vs `descriptor_version`) lets a descriptor-algorithm bump re-analyse just the descriptor columns without re-running TensorFlow on every track. The worker pool honours that distinction with two job types.

## Scaling

- Brute-force similarity stays comfortable up to ~250k tracks. Above that, swap `harmonie.similarity` for FAISS or hnswlib — the API is small.
- 100k tracks × 1280 floats = 512 MB of embeddings in memory plus the model and overhead. Expect 1.5–2 GB RSS.
- Initial scans of large libraries are CPU-bound at ~4 s/track. Scale `HARMONIE_WORKERS` to your core count; each worker holds its own copy of the model (~200 MB).
- SQLite is fine to several million rows. Move to Postgres only if you want multiple service instances sharing one DB.

## Development

```bash
pip install -e ".[dev]"
pytest
```

The non-Essentia parts (DB, similarity, playlist, scan, tags) are covered by the `tests/` suite. Essentia itself is exercised by running an actual scan.

## Schema migrations

The schema lives in `harmonie/migrations.py` as an append-only list of versioned migration functions. The runner is invoked automatically when the DB is opened (so `harmonie scan`, `harmonie serve`, and any test that constructs a `Database` always sees an up-to-date schema), and is also exposed as a one-shot CLI for ops:

```bash
harmonie migrate
# Migrated database from version 0 to 1.
harmonie migrate
# Already at schema version 1. Nothing to do.
```

Each migration runs in its own transaction. A failure mid-migration rolls back to the previous version and leaves the DB usable. The runner refuses to operate on a DB that has been migrated past the latest version known to this binary — it would risk silent data loss.

To add a migration:

1. Add a `_migration_NNN_what_it_does(conn)` function to `harmonie/migrations.py`.
2. Append it to the `MIGRATIONS` list in the same file.
3. Update any code in `db.py` that depends on the new shape — column lists in `upsert_track`, `list_tracks`, `get_tracks_by_ids`, etc.

Migration functions must use `conn.execute(...)` for each statement individually rather than `conn.executescript(...)`, because the latter issues its own commit and breaks the surrounding transaction.

## Continuous integration

Two workflows run on every push and pull request:

* **[`tests.yml`](.github/workflows/tests.yml)** — `pytest` against Python 3.9 and 3.11.
* **[`docker.yml`](.github/workflows/docker.yml)** — `docker buildx` of the production image for `linux/amd64`. On every push to `main` the image is published to `ghcr.io/mxschll/harmonie` with `latest`, `main`, `sha-<short>`, and an auto-incrementing `0.1.<n>` version tag (where `n` is the workflow run number — no manual tagging required). On a `vX.Y.Z` git tag, the image picks up `X.Y.Z`, `X.Y`, and `X` as well. On pull requests the image is built but not pushed.

## Layout

```
harmonie/
├── harmonie/
│   ├── api/           # FastAPI app, routes, schemas
│   ├── analyzer.py    # scan orchestration + scheduler; owns DB and index
│   ├── cli.py         # argparse CLI
│   ├── config.py      # pydantic-settings + logging
│   ├── db.py          # SQLite layer
│   ├── features.py    # Essentia extraction
│   ├── index.py       # in-memory L2-normalised embedding cache
│   ├── migrations.py  # versioned schema migrations
│   ├── playlist.py    # similar / chained / vibe playlists, Camelot wheel
│   ├── scan.py        # filesystem walker; library-relative path helper
│   ├── similarity.py  # cosine search (thin layer over EmbeddingIndex)
│   ├── tags.py        # mutagen-based tag extraction
│   └── workers.py     # multiprocessing.Pool wrapper
├── tests/
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```
