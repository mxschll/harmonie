# harmonie

[![Tests (Python 3.9)](https://github.com/mxschll/harmonie/actions/workflows/tests-py39.yml/badge.svg)](https://github.com/mxschll/harmonie/actions/workflows/tests-py39.yml)
[![Tests (Python 3.11)](https://github.com/mxschll/harmonie/actions/workflows/tests-py311.yml/badge.svg)](https://github.com/mxschll/harmonie/actions/workflows/tests-py311.yml)
[![Lint](https://github.com/mxschll/harmonie/actions/workflows/lint.yml/badge.svg)](https://github.com/mxschll/harmonie/actions/workflows/lint.yml)

Harmonie provides audio similarity and playlist generation for local music libraries. It analyzes each track with [Essentia](https://essentia.upf.edu/) to produce an audio embedding, musical descriptors, and probabilities for [400 Discogs styles](https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.json). [Mutagen](https://mutagen.readthedocs.io/) reads artist, album, title, and track-number tags.

Results are stored in SQLite and exposed through an HTTP API for similarity search, filtered track listings, and playlist generation. Harmonie is the analysis service used by the [Jellyfin Harmonie plugin](https://github.com/mxschll/jellyfin-harmonie).

## Requirements

- Python 3.9–3.12; `essentia-tensorflow` does not provide Python 3.13 wheels
- x86_64 or Apple Silicon
- Approximately 1 GB of RAM per analysis worker

## Contents

- [Install](#install)
- [API](#api) (full reference in [API.md](API.md))
- [Configuration](#configuration)
- [Running as a service](#running-as-a-service)
- [Development](#development)

## Install

[pipx][pipx] installs Harmonie in an isolated environment and adds the `harmonie` command to your `PATH`. The `--pre` flag is required because `essentia-tensorflow` is published as a development release.

If your system Python is 3.9–3.12:

```bash
sudo apt install pipx
pipx ensurepath
pipx install --pip-args='--pre' 'git+https://github.com/mxschll/harmonie.git'
```

If your system Python is 3.13+, grab Python 3.12 via [uv][uv] and point pipx at it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
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

On first start, Harmonie scans the complete library. Analysis takes about one second per track on a modern x86 core and scales with `HARMONIE_WORKERS` until limited by storage throughput. Subsequent scans only reanalyze files whose size or modification time changed.

Watch progress and trigger another scan:

```bash
curl http://localhost:8842/api/v1/scan | json_pp
curl -X POST http://localhost:8842/api/v1/scan | json_pp
```

[pipx]: https://pipx.pypa.io/
[uv]: https://docs.astral.sh/uv/

## API

Harmonie exposes its HTTP API under `/api/v1/`. See [API.md](API.md) for endpoints, request fields, and examples.

## Configuration

All settings come from environment variables (or a `.env` file in the working directory):

| Variable | Default | Purpose |
| --- | --- | --- |
| `HARMONIE_LIBRARIES` | (none) | Comma- or colon-separated absolute paths to scan |
| `HARMONIE_DATA_DIR` | platform user-data dir | Where to put `harmonie.db`. Defaults to `~/.local/share/harmonie` on Linux, `~/Library/Application Support/harmonie` on macOS. |
| `HARMONIE_WORKERS` | CPU count | Analysis worker processes. Each worker can use about 1 GB of RAM. |
| `HARMONIE_SCAN_INTERVAL_HOURS` | `24` | Periodic scan interval (`0` disables) |
| `HARMONIE_SCAN_ON_STARTUP` | `true` | Run a scan immediately on boot |
| `HARMONIE_HOST` | `0.0.0.0` | HTTP bind address |
| `HARMONIE_PORT` | `8842` | HTTP port |
| `HARMONIE_API_KEY` | (none) | If set, required in `X-API-Key` |
| `HARMONIE_CORS_ORIGINS` | (none) | Comma-separated list to enable CORS |
| `HARMONIE_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

## Running as a service

For long-lived Linux deployments, run Harmonie under systemd.

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
journalctl -u harmonie -f
```

After config changes, `sudo systemctl restart harmonie`.

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for architecture, scaling notes, the test workflow, and schema migration guidance.
