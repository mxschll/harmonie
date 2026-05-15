# harmonie

Audio similarity service. Scans a music library, extracts a per-track embedding plus musical descriptors (BPM, key, loudness, danceability, onset rate) and reads the file's tags (artist, album, title, track number) using [Essentia](https://essentia.upf.edu/) and [mutagen](https://mutagen.readthedocs.io/), stores everything in SQLite, and exposes an HTTP API for similarity queries and playlist generation.

The default backend is Essentia's **Discogs-Effnet** model — a 1280-d embedding trained on Discogs tags that's well suited to music similarity. A lighter `MusicExtractor` backend is available for hosts without TensorFlow.

The intended deployment is a long-running container that periodically rescans the library and serves any HTTP client — a media-server plugin, a custom playlist generator, a CLI tool — that wants similarity queries against the indexed catalog.

## Quick start (Docker)

Pre-built multi-arch images (`linux/amd64` + `linux/arm64`) are published to GitHub Container Registry on every push to `main` and on each version tag:

```bash
docker pull ghcr.io/mxschll/harmonie:latest
```

Available tags:

| Tag                  | What it tracks                 |
| -------------------- | ------------------------------ |
| `latest`             | Latest commit on `main`        |
| `main`               | Same as `latest`               |
| `vX.Y.Z` / `X.Y` / `X` | Released versions (git tags) |
| `sha-<short>`        | A specific commit              |

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
| `POST` | `/api/v1/playlists/similar` | N-track playlist from seeds, with BPM/key constraints |
| `POST` | `/api/v1/playlists/chained` | Walk top-N similar in chunks, re-anchoring on the last track each chunk |
| `POST` | `/api/v1/playlists/vibe` | N-track playlist from descriptor targets |

OpenAPI docs are served at `/docs` (Swagger UI) and `/openapi.json`.

### Mapping harmonie tracks to an external catalog

Every track and every match in API responses includes the metadata you need to look it up in another system without doing a filesystem walk.

* **`artist` + `album` + `title` + `track_number`** — the tag-based match. The four fields together are usually enough to identify a track unambiguously in another catalog.
* **`library_root` + `relative_path`** — the path-based match. If the consumer sees the same library layout under a different mount point, it joins on `relative_path` directly. No path-prefix mapping config needed in the common case.

`library_root` reflects the configured `HARMONIE_LIBRARIES` entries at scan time. If you reconfigure mount points, re-scan to refresh.

### Filter parameters

Both `/tracks` and `/tracks/{id}/similar` accept the same set of optional descriptor filters as query params:

```
bpm_min, bpm_max, key (repeatable), scale,
danceability_min, danceability_max, loudness_min, loudness_max
```

For playlist endpoints the same set is in the body under `filter`.

### Playlists

```bash
# Similar to a seed, harmonically compatible (Camelot wheel), max ±5 BPM jump.
curl -X POST http://localhost:8842/api/v1/playlists/similar \
  -H 'content-type: application/json' \
  -d '{"seed_ids": [1], "n": 20, "bpm_drift": 5, "harmonic_mix": true}'

# Vibe: 128 BPM target, danceability >= 1.5, 30 tracks.
curl -X POST http://localhost:8842/api/v1/playlists/vibe \
  -H 'content-type: application/json' \
  -d '{"n": 30, "target_bpm": 128, "filter": {"danceability_min": 1.5}}'

# Chained walk: 5 similar to the seed, then 5 similar to that chunk's last
# track, repeat until 25 tracks. No track ever appears twice.
curl -X POST http://localhost:8842/api/v1/playlists/chained \
  -H 'content-type: application/json' \
  -d '{"seed_id": 1, "chunk_size": 5, "n": 25}'
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

## Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request:

* `pytest` against Python 3.9 and 3.11.
* `docker buildx` of the production image for `linux/amd64` and `linux/arm64`.
* On pushes to `main` or version tags (`vX.Y.Z`), the image is published to `ghcr.io/mxschll/harmonie` with `latest`, `sha-<short>`, and semver tags.
* On pull requests the image is built but not pushed — Dockerfile changes are validated without touching the registry.

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
