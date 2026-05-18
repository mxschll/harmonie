# harmonie

[![Tests (Python 3.9)](https://github.com/mxschll/harmonie/actions/workflows/tests-py39.yml/badge.svg)](https://github.com/mxschll/harmonie/actions/workflows/tests-py39.yml)
[![Tests (Python 3.11)](https://github.com/mxschll/harmonie/actions/workflows/tests-py311.yml/badge.svg)](https://github.com/mxschll/harmonie/actions/workflows/tests-py311.yml)
[![Lint](https://github.com/mxschll/harmonie/actions/workflows/lint.yml/badge.svg)](https://github.com/mxschll/harmonie/actions/workflows/lint.yml)

Audio similarity service. Scans a music library, extracts a per-track embedding plus musical descriptors (BPM, key, loudness, danceability, onset rate), classifies each track against the 400 Discogs styles (House, Techno, Trap, Punk, ...), and reads file tags (artist, album, title, track number) using [Essentia](https://essentia.upf.edu/) and [mutagen](https://mutagen.readthedocs.io/). Everything is stored in SQLite and exposed via an HTTP API for similarity queries, style-filtered listings, and playlist generation.

Built around Essentia's Discogs-Effnet model — a 1280-d embedding trained on Discogs tags that captures genre / style / sonic character.

[Run it as a long-lived service](#running-as-a-service). It rescans on a schedule and serves any HTTP client that wants similarity queries against the indexed catalog.

## Requirements

* Linux or macOS
* Python 3.9–3.12 (the upstream `essentia-tensorflow` wheels don't cover 3.13+ yet)
* x86_64 CPU with AVX (essentially anything from the last decade) or Apple Silicon
* `~1 GB` RAM per worker process during extraction; budget accordingly

## Contents

- [Install](#install)
- [API](#api) (full reference in [API.md](API.md))
- [Configuration](#configuration)
- [Running as a service](#running-as-a-service)

## Install

[pipx][pipx] puts harmonie in its own isolated virtualenv with the `harmonie` binary on your PATH. The `--pre` flag is needed because `essentia-tensorflow` is published as a `.dev` pre-release.

If your system Python is 3.9–3.12:

```bash
sudo apt install pipx
pipx ensurepath
pipx install --pip-args='--pre' 'git+https://github.com/mxschll/harmonie.git'
```

If your system Python is 3.13+ (Ubuntu 24.10+, current Fedora, etc.), grab Python 3.12 via [uv][uv] and point pipx at it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # one-time
uv python install 3.12
pipx install --python "$(uv python find 3.12)" --pip-args='--pre' \
  'git+https://github.com/mxschll/harmonie.git'
```

Run it:

```bash
HARMONIE_LIBRARIES=/path/to/music harmonie serve
```

Update with `pipx upgrade harmonie`.

### First scan

On first start, harmonie scans your library. Effnet inference is about a second per track on a modern x86 core. Throughput scales roughly with `HARMONIE_WORKERS` until storage I/O becomes the bottleneck — expect 1–2 seconds per track on local SSDs with sane worker counts, 10×+ slower on network filesystems. A 50k-track library on a fast box takes around a day; the same library on a slow CPU or remote mount can take several. Subsequent scans are incremental: only files whose size or mtime changed get re-extracted.

Watch progress and trigger another scan:

```bash
curl http://localhost:8842/api/v1/scan | json_pp
curl -X POST http://localhost:8842/api/v1/scan | json_pp
```

[pipx]: https://pipx.pypa.io/
[uv]: https://docs.astral.sh/uv/

## API

harmonie exposes an HTTP API under `/api/v1/` for similarity queries, style-filtered listings, and playlist generation. See [API.md](API.md) for the full reference: endpoints, filter syntax, style filtering, playlist modes, and end-to-end examples.

`GET /health` is always public; if `HARMONIE_API_KEY` is set, all `/api/v1/` endpoints require `X-API-Key: <key>`.

## Configuration

All settings come from environment variables (or a `.env` file in the working directory):

| Variable | Default | Purpose |
| --- | --- | --- |
| `HARMONIE_LIBRARIES` | (none) | Comma- or colon-separated absolute paths to scan |
| `HARMONIE_DATA_DIR` | platform user-data dir | Where to put `harmonie.db`. Defaults to `~/.local/share/harmonie` on Linux, `~/Library/Application Support/harmonie` on macOS. |
| `HARMONIE_WORKERS` | CPU count | Analysis worker processes. Each worker peaks around 1 GB of RAM during extraction of long files (10+ minute classical recordings, etc.), so set this conservatively on RAM-constrained boxes. |
| `HARMONIE_SCAN_INTERVAL_HOURS` | `6` | Periodic scan interval (`0` disables) |
| `HARMONIE_SCAN_ON_STARTUP` | `true` | Run a scan immediately on boot |
| `HARMONIE_HOST` | `0.0.0.0` | HTTP bind address |
| `HARMONIE_PORT` | `8842` | HTTP port |
| `HARMONIE_API_KEY` | (none) | If set, required in `X-API-Key` |
| `HARMONIE_CORS_ORIGINS` | (none) | Comma-separated list to enable CORS |
| `HARMONIE_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |


## Running as a service

For long-lived deployments, run harmonie under systemd rather than in a shell or tmux. Systemd handles restarts, captures logs, and isn't tied to a login session.

Create `/etc/systemd/system/harmonie.service`:

```ini
[Unit]
Description=harmonie audio similarity service
After=network-online.target

[Service]
Type=exec
User=<your-user>
WorkingDirectory=/home/<your-user>
Environment=HARMONIE_LIBRARIES=/path/to/music
Environment=HARMONIE_DATA_DIR=/home/<your-user>/.local/share/harmonie
Environment=HARMONIE_WORKERS=6
Environment=HARMONIE_PORT=8842
ExecStart=/home/<your-user>/.local/bin/harmonie serve
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Replace `<your-user>`, paths, and env vars with your own. Then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now harmonie
sudo systemctl status harmonie
journalctl -u harmonie -f         # live logs
```

After config changes, `sudo systemctl restart harmonie`.


## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for architecture, scaling notes, the test workflow, and schema migration guidance.
