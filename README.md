# harmonie

[![Tests (Python 3.9)](https://github.com/mxschll/harmonie/actions/workflows/tests-py39.yml/badge.svg)](https://github.com/mxschll/harmonie/actions/workflows/tests-py39.yml)
[![Tests (Python 3.11)](https://github.com/mxschll/harmonie/actions/workflows/tests-py311.yml/badge.svg)](https://github.com/mxschll/harmonie/actions/workflows/tests-py311.yml)

Audio similarity service. Scans a music library, extracts a per-track embedding plus musical descriptors (BPM, key, loudness, danceability, onset rate), classifies each track against the 400 Discogs styles (House, Techno, Trap, Punk, ...), and reads file tags (artist, album, title, track number) using [Essentia](https://essentia.upf.edu/) and [mutagen](https://mutagen.readthedocs.io/). Everything is stored in SQLite and exposed via an HTTP API for similarity queries, style-filtered listings, and playlist generation.

The default backend is Essentia's Discogs-Effnet model (1280-d embedding trained on Discogs tags). A lighter `MusicExtractor` backend is available for hosts without TensorFlow.

Run it as a long-lived service. It rescans on a schedule and serves any HTTP client that wants similarity queries against the indexed catalog.

## Contents

- [Install](#install)
- [API](#api)
- [Playlists](#playlists)
- [Configuration](#configuration)

## Install

The fastest install on any host with Python 3.9+ is via [pipx][pipx], which puts harmonie in its own isolated virtualenv and the `harmonie` binary on your PATH.

```bash
# One-time: install pipx itself.
sudo apt install pipx           # Debian/Ubuntu
# sudo dnf install pipx         # Fedora
# brew install pipx             # macOS
# python3 -m pip install --user pipx   # any other distro
pipx ensurepath

# Install harmonie from GitHub. --pre is required because essentia-tensorflow
# is published as a .dev pre-release.
pipx install --pip-args='--pre' 'git+https://github.com/mxschll/harmonie.git'

HARMONIE_LIBRARIES=/path/to/music harmonie serve
```

Update with one command:

```bash
pipx upgrade harmonie
```

### First scan

On first start, harmonie scans your library. Effnet inference is about a second per track on a fast core, longer on slow CPUs or network mounts. Large libraries on slow disks can take a day.

Watch progress:

```bash
curl http://localhost:8842/api/v1/scan
```

Trigger another scan:

```bash
curl -X POST http://localhost:8842/api/v1/scan
```

Subsequent scans are incremental. Only files whose size or mtime changed get re-extracted.

[pipx]: https://pipx.pypa.io/

## API

All endpoints are versioned under `/api/v1/`. If `HARMONIE_API_KEY` is set, every authenticated request must include `X-API-Key: <key>`. `GET /health` is always public.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET`  | `/health` | Liveness probe |
| `GET`  | `/api/v1/status` | Service overview: version, libraries, model, versions, track counts, duration, db size, by-model |
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

Both URL and body forms accept the same filters and produce the same internal `TrackFilter`. Pick whichever shape fits the call site.

URL form (`/tracks`, `/tracks/{id}/similar`):

```
?bpm=120..130        closed range
?bpm=120..           lower bound only
?bpm=..130           upper bound only
?bpm=128             exact value
?key=A&key=B         set membership (repeat the parameter)
?style=Electronic    prefix match: every Electronic--* style
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

Track and match responses include the metadata you need to look a track up in another system without doing a filesystem walk.

* `artist`, `album`, `title`, `track_number`: the tag-based match. The four fields together are usually enough to identify a track unambiguously.
* `library_root`, `relative_path`: the path-based match. If the consumer sees the same library layout under a different mount point, it joins on `relative_path` directly. No path-prefix mapping config in the common case.

`library_root` reflects the configured `HARMONIE_LIBRARIES` entries at scan time. If you reconfigure mount points, re-scan to refresh.

`GET /api/v1/tracks/resolve` exposes this matching directly. Pass any subset of `{path, artist, album, title}` as query params. The endpoint runs a multi-strategy ladder (exact path → relative path → full tag triple → looser tag pair, all NOCASE for tags) and returns the first hit (smallest id wins on ties):

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

# Prefix match: every Electronic style.
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

`POST /api/v1/playlists` builds every kind of playlist. The body has a required `mode` field that selects the strategy. Each mode has its own validated schema; there are no hidden parameter coupling rules.

**Picking a mode:**

| Use case | Mode |
| --- | --- |
| "More tracks like this one" | `similar` with one seed |
| "More like these few" | `similar` with multiple seeds |
| "An endless radio" | `similar`, then re-seed with the last few items |
| "A long mix that gradually changes style" | `drift` |
| "Tracks at ~128 BPM, danceable, electronic, shuffled" | `vibe` with `filter` + `target` |

#### Mode `similar`: track radio

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

**Endless radio.** The endpoint returns a fixed `n`. To keep going, re-seed from the tail of the previous response:

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

#### Mode `drift`: chunked walk

`drift` walks gradually away from one seed. Each chunk of `chunk_size` tracks is anchored on the last pick, so the playlist evolves in style as it goes.

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

Tuning `chunk_size`:

* `1`: every new track becomes the next anchor. Drifts the fastest.
* `5` (default): re-anchors every five picks. Signature drift behaviour.
* `20`: re-anchors rarely. Stays close to the seed for most of the playlist.
* `n` (= total length): equivalent to `similar` mode (no re-anchoring).

Scores typically jump at chunk boundaries because the first track of each chunk is measured against a new anchor, not the original seed.

#### Mode `vibe`: descriptor-driven

No seeds. The `filter` block narrows the candidate pool; the `target` block ranks within it by closeness.

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
| `filter` | all | none | | Hard candidate-pool constraints. Same shape as the URL filter, in body form. |
| `seeds` | similar, drift | required | similar: ≥1, drift: exactly 1 | Track IDs to anchor on. |
| `include_seeds` | similar, drift | `false` | | Include the seed track(s) in the result. |
| `smooth_transitions.bpm_tolerance` | similar, drift | `null` | ≥0 | Max BPM gap between consecutive picks. Lenient on missing BPMs. |
| `smooth_transitions.key_compatible` | similar, drift | `false` | | Restrict consecutive picks to harmonically compatible keys (Camelot wheel: same key, ±1 number, parallel mode). Strict: tracks without key info are dropped. |
| `chunk_size` | drift | `5` | 1–100 | Tracks per anchor before re-anchoring on the last pick. Larger stays closer to the seed; smaller drifts faster. |
| `target.bpm` | vibe | `null` | >0 | Soft preference. Tracks closer to this BPM rank higher. |
| `target.danceability` | vibe | `null` | ≥0 | Soft preference for closeness to this danceability score. |
| `shuffle` | vibe | `true` | | Randomise the (post-target) pool before truncation. |
| `rng_seed` | vibe | `null` | | Seed for reproducible shuffling. `null` is fresh randomness each call. |

The bare-minimum body for `similar`/`drift` (`mode` and `seeds`) returns 20 tracks with no BPM/key constraints, seeds excluded from output, and (for drift) chunks of 5.

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

# Resolve a track by tags, then ask for 20 similar with key compatibility.
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
| `HARMONIE_DATA_DIR` | platform user-data dir | Where to put `harmonie.db`. Defaults to `~/.local/share/harmonie` on Linux, `~/Library/Application Support/harmonie` on macOS. |
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


## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for architecture, scaling notes, the test workflow, and schema migration guidance.
