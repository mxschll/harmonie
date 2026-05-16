# harmonie

[![Tests (Python 3.9)](https://github.com/mxschll/harmonie/actions/workflows/tests-py39.yml/badge.svg)](https://github.com/mxschll/harmonie/actions/workflows/tests-py39.yml)
[![Tests (Python 3.11)](https://github.com/mxschll/harmonie/actions/workflows/tests-py311.yml/badge.svg)](https://github.com/mxschll/harmonie/actions/workflows/tests-py311.yml)
[![Docker](https://github.com/mxschll/harmonie/actions/workflows/docker.yml/badge.svg)](https://github.com/mxschll/harmonie/actions/workflows/docker.yml)

Audio similarity service. Scans a music library, extracts a per-track embedding plus musical descriptors (BPM, key, loudness, danceability, onset rate), classifies each track against the 400 Discogs styles (House, Techno, Trap, Punk, …), and reads the file's tags (artist, album, title, track number) using [Essentia](https://essentia.upf.edu/) and [mutagen](https://mutagen.readthedocs.io/). Everything is stored in SQLite, and exposed via an HTTP API for similarity queries, style-filtered listings, and playlist generation.

The default backend is Essentia's **Discogs-Effnet** model — a 1280-d embedding trained on Discogs tags that's well suited to music similarity. A lighter `MusicExtractor` backend is available for hosts without TensorFlow.

The intended deployment is a long-running container that periodically rescans the library and serves any HTTP client — a media-server plugin, a custom playlist generator, a CLI tool — that wants similarity queries against the indexed catalog.

## Quick start (Docker)

A `linux/amd64` image is published to GitHub Container Registry on every push to `main` and on each version tag:

```bash
# x86_64 hosts (most cloud Linux, Intel Macs):
docker pull ghcr.io/mxschll/harmonie:latest

# arm64 hosts (Apple Silicon, Pi 4/5, AWS Graviton, …) — pulling without
# --platform fails with "no matching manifest". Force the amd64 image and
# Docker runs it via emulation (Rosetta on Apple Silicon, qemu elsewhere):
docker pull --platform linux/amd64 ghcr.io/mxschll/harmonie:latest
docker run  --platform linux/amd64 --rm ghcr.io/mxschll/harmonie:latest

# Or set it once for the shell so you don't have to keep typing it:
export DOCKER_DEFAULT_PLATFORM=linux/amd64
```

> The image is amd64-only because Essentia ships only manylinux x86_64 wheels on PyPI. Apple Silicon runs amd64 images well via Rosetta (~30–50% slower than native for Essentia inference); slower arm64 hosts that can't tolerate emulation can install directly with `pip` (see the local-Python instructions below).

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

All endpoints are versioned under `/api/v1/`. If `HARMONIE_API_KEY` is set, every authenticated request must include `X-API-Key: <key>`. `GET /health` is always public.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET`  | `/health` | Liveness probe |
| `GET`  | `/api/v1/info` | Static service info: version, libraries, model, schema and descriptor versions |
| `GET`  | `/api/v1/stats` | Dynamic counters: track count, duration, db size, by-model |
| `GET`  | `/api/v1/scan` | Current scan state and counters |
| `POST` | `/api/v1/scan` | Trigger a scan now (`?force=true` to ignore mtime/size) |
| `GET`  | `/api/v1/tracks` | List tracks (filter + pagination) |
| `GET`  | `/api/v1/tracks/{id}` | Full track record |
| `GET`  | `/api/v1/tracks/{id}/similar` | Top-N similar tracks |
| `GET`  | `/api/v1/tracks/resolve` | Find one track by `path` and/or tags |
| `GET`  | `/api/v1/styles` | Enumerate Discogs-400 styles in the library |
| `POST` | `/api/v1/playlists` | Build a playlist (mode set explicitly in the body) |

OpenAPI docs are served at `/docs` (Swagger UI) and `/openapi.json`.

### Filters

Both URL and body forms accept the same set of filters and produce the same internal `TrackFilter`. Pick whichever shape fits the call site.

URL form (`/tracks`, `/tracks/{id}/similar`):

```
?bpm=120..130        closed range
?bpm=120..           lower bound only
?bpm=..130           upper bound only
?bpm=128             exact value
?key=A&key=B         set membership (repeat the parameter)
?style=Electronic    prefix match — every Electronic--* style
?style=Electronic---House  exact label
?style_min=0.5       only count style rows above this probability
?style_mode=all      tracks must match every requested style (default: any)
```

Available filter fields: `bpm`, `danceability`, `loudness`, `key`, `scale`, `style`, `style_min`, `style_mode`.

Body form (`POST /playlists` under `filter`):

```json
{
  "filter": {
    "bpm":      { "gte": 120, "lte": 130 },
    "loudness": { "lte": -10 },
    "key":      ["A", "B"],
    "scale":    "minor",
    "style":    ["Electronic"],
    "style_min": 0.5,
    "style_mode": "any"
  }
}
```

### Mapping harmonie tracks to an external catalog

Every track and every match in API responses includes the metadata you need to look it up in another system without doing a filesystem walk.

* **`artist` + `album` + `title` + `track_number`** — the tag-based match. The four fields together are usually enough to identify a track unambiguously in another catalog.
* **`library_root` + `relative_path`** — the path-based match. If the consumer sees the same library layout under a different mount point, it joins on `relative_path` directly. No path-prefix mapping config needed in the common case.

`library_root` reflects the configured `HARMONIE_LIBRARIES` entries at scan time. If you reconfigure mount points, re-scan to refresh.

`GET /api/v1/tracks/resolve` exposes this matching directly. Pass any subset of `{path, artist, album, title}` as query params; the endpoint runs a multi-strategy ladder (exact path → relative path → full tag triple → looser tag pair, all NOCASE for tags) and returns the first hit (smallest id wins on ties):

```bash
# By tags.
curl --get http://localhost:8842/api/v1/tracks/resolve \
  --data-urlencode 'artist=Aphex Twin' \
  --data-urlencode 'album=SAW' \
  --data-urlencode 'title=Xtal'

# By path (works against the absolute path or relative_path).
curl --get http://localhost:8842/api/v1/tracks/resolve \
  --data-urlencode 'path=Aphex Twin/SAW/01 Xtal.flac'
```

400 on an empty request, 404 if no strategy matches.

### Styles

During scan, harmonie runs Essentia's Discogs-400 classifier head on the same Effnet embeddings used for similarity. Each track gets a 400-dimensional probability vector over Discogs styles like `Electronic---House`, `Hip Hop---Trap`, or `Rock---Punk`. The top 10 (and any above 5% probability) are stored as filterable rows; the full vector is kept as a BLOB for clustering.

```bash
# Filter tracks by exact style.
curl 'http://localhost:8842/api/v1/tracks?style=Electronic---House'

# Prefix match — every Electronic style.
curl --get 'http://localhost:8842/api/v1/tracks' --data-urlencode 'style=Electronic'

# Multiple styles. Default match is "any"; use style_mode=all for AND.
curl --get 'http://localhost:8842/api/v1/tracks' \
  --data-urlencode 'style=Electronic---House' \
  --data-urlencode 'style=Electronic---Techno'

# Demand confidence: only count style rows above 0.5 probability.
curl 'http://localhost:8842/api/v1/tracks?style=Electronic&style_min=0.5'

# Enumerate styles present in the library.
curl 'http://localhost:8842/api/v1/styles?style_min=0.5'
```

### Playlists

`POST /api/v1/playlists` builds every kind of playlist. The body has a required `mode` field that selects the strategy. Each mode has its own validated schema — there are no hidden parameter coupling rules.

**Picking a mode:**

| Use case | Mode |
| --- | --- |
| "More tracks like this one" | `similar` with one seed |
| "More like these few" | `similar` with multiple seeds |
| "An endless radio" | `similar`, then re-seed with the last few items |
| "A long mix that gradually changes style" | `drift` |
| "Tracks at ~128 BPM, danceable, electronic, shuffled" | `vibe` with `filter` + `target` |

#### Mode `similar` — track radio

The seeds anchor the playlist; results stay close to their embedding centroid. This is the "Track Radio" surface.

```bash
# Minimum: 20 tracks similar to track 42.
curl -X POST http://localhost:8842/api/v1/playlists \
  -H 'content-type: application/json' \
  -d '{"mode": "similar", "seeds": [42]}'

# Tighter: multi-seed, smooth transitions, hard filter, include the seeds.
curl -X POST http://localhost:8842/api/v1/playlists \
  -H 'content-type: application/json' \
  -d '{
    "mode": "similar",
    "seeds": [42, 117],
    "n": 30,
    "smooth_transitions": { "bpm_tolerance": 5, "key_compatible": true },
    "filter": { "bpm": { "gte": 120, "lte": 140 }, "style_min": 0.3 },
    "include_seeds": true
  }'
```

**Endless radio** — the endpoint returns a fixed `n`. To keep going, re-seed from the tail of the previous response:

```bash
seed=$(curl -sX POST http://localhost:8842/api/v1/playlists \
  -H 'content-type: application/json' \
  -d '{"mode":"similar","seeds":[42],"n":20}' \
  | jq '[.items[-3:][].track_id]')
# Next batch is "music like the last 3 tracks of the previous batch."
curl -X POST http://localhost:8842/api/v1/playlists \
  -H 'content-type: application/json' \
  -d "{\"mode\":\"similar\",\"seeds\":$seed,\"n\":20}"
```

#### Mode `drift` — chunked walk

`drift` walks gradually away from one seed. Each chunk of `chunk_size` tracks is anchored on the last pick, so the playlist evolves in style as it goes:

```bash
curl -X POST http://localhost:8842/api/v1/playlists \
  -H 'content-type: application/json' \
  -d '{
    "mode": "drift",
    "seeds": [42],
    "n": 30,
    "chunk_size": 5,
    "smooth_transitions": { "key_compatible": true }
  }'
```

Tuning intuition for `chunk_size`:

* `1` — every new track becomes the next anchor. Drifts the fastest.
* `5` (default) — moderate. Re-anchors every five picks. Signature drift behaviour.
* `20` — re-anchors rarely. Stays close to the seed for most of the playlist.
* `n` (= total length) — equivalent to `similar` mode (no re-anchoring at all).

You'll see the drift visibly: scores typically jump at chunk boundaries because the first track of each chunk is measured against a *new* anchor, not the original seed.

#### Mode `vibe` — descriptor-driven

No seeds. The `filter` block narrows the candidate pool; the `target` block ranks within it by closeness:

```bash
curl -X POST http://localhost:8842/api/v1/playlists \
  -H 'content-type: application/json' \
  -d '{
    "mode": "vibe",
    "n": 30,
    "filter": { "bpm": { "gte": 120, "lte": 130 } },
    "target": { "bpm": 128, "danceability": 1.5 },
    "shuffle": true,
    "rng_seed": 42
  }'
```

#### Body field reference

| Field | Modes | Default | Range | Purpose |
| --- | --- | --- | --- | --- |
| `mode` | all | required | `similar` \| `drift` \| `vibe` | Strategy selector. |
| `n` | all | `20` | 1–500 | Number of tracks to return. |
| `filter` | all | none | — | Hard candidate-pool constraints — same shape as the URL filter, in body form. |
| `seeds` | similar, drift | required | similar: ≥1, drift: exactly 1 | Track IDs to anchor on. |
| `include_seeds` | similar, drift | `false` | — | Include the seed track(s) in the result. |
| `smooth_transitions.bpm_tolerance` | similar, drift | `null` | ≥0 | Max BPM gap between consecutive picks. Lenient on missing BPMs. |
| `smooth_transitions.key_compatible` | similar, drift | `false` | — | Restrict consecutive picks to harmonically compatible keys (Camelot wheel: same key, ±1 number, parallel mode). Strict — tracks without key info are dropped. |
| `chunk_size` | drift | `5` | 1–100 | Tracks per anchor before re-anchoring on the last pick. Larger = stays closer to the seed; smaller = drifts faster. |
| `target.bpm` | vibe | `null` | >0 | Soft preference — tracks closer to this BPM rank higher. |
| `target.danceability` | vibe | `null` | ≥0 | Soft preference for closeness to this danceability score. |
| `shuffle` | vibe | `true` | — | Randomise the (post-target) pool before truncation. |
| `rng_seed` | vibe | `null` | — | Seed for reproducible shuffling. `null` = fresh randomness each call. |

If you call `POST /playlists` with the bare minimum body for `similar`/`drift` (`mode` and `seeds`), you get 20 tracks, no BPM/key constraints, seeds excluded from output, and (for drift) chunks of 5. That's a sensible starting point — add the smooth-transition and filter blocks when the surface needs them.

### Cross-cutting examples

```bash
# Trigger a scan and watch its progress.
curl -X POST 'http://localhost:8842/api/v1/scan?force=true'
while [ "$(curl -sS http://localhost:8842/api/v1/scan | jq -r .state)" != "idle" ]; do
  sleep 5
done

# Find every Hard Techno track at 140+ BPM, sorted by BPM ascending.
curl --get http://localhost:8842/api/v1/tracks \
  --data-urlencode 'style=Electronic---Hard Techno' \
  --data 'bpm=140..' --data 'order_by=bpm'

# Resolve a track from a Spotify-imported playlist by tags, then ask for
# 20 similar with key compatibility.
id=$(curl --get http://localhost:8842/api/v1/tracks/resolve \
  --data-urlencode 'artist=Aphex Twin' \
  --data-urlencode 'title=Xtal' | jq .id)
curl -X POST http://localhost:8842/api/v1/playlists \
  -H 'content-type: application/json' \
  -d "{\"mode\":\"similar\",\"seeds\":[$id],\"n\":20,
       \"smooth_transitions\":{\"key_compatible\":true}}"
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

Three workflows run on every push and pull request:

* **[`tests-py39.yml`](.github/workflows/tests-py39.yml)** — `pytest` against Python 3.9 (the declared minimum).
* **[`tests-py311.yml`](.github/workflows/tests-py311.yml)** — `pytest` against Python 3.11 (matches the production Dockerfile).
* **[`docker.yml`](.github/workflows/docker.yml)** — `docker buildx` of the production image for `linux/amd64`. On every push to `main` the image is published to `ghcr.io/mxschll/harmonie` with `latest`, `main`, `sha-<short>`, and an auto-incrementing `0.1.<n>` version tag (where `n` is the workflow run number — no manual tagging required). On a `vX.Y.Z` git tag, the image picks up `X.Y.Z`, `X.Y`, and `X` as well. On pull requests the image is built but not pushed.

Splitting the test workflow per Python version means the badges at the top of this README show each version's status independently — a regression on one Python doesn't make both badges go red.

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
