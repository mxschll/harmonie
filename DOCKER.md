> [!NOTE]
> The image requires an x86-64 host with AVX (Intel Sandy Bridge / AMD
> Bulldozer, 2011 or newer). See
> [CPU, architecture, and GPUs](#cpu-architecture-and-gpus).

## Run it

```bash
MUSIC_DIR=/path/to/music docker compose up -d
```

Harmonie listens on port 8842 and starts its first scan immediately. Follow it
with `docker compose logs -f`, or from the Jellyfin plugin's settings page.

With plain Docker:

```bash
docker run -d --name harmonie \
  -p 8842:8842 \
  -v ./harmonie-data:/data \
  -v /path/to/music:/music:ro \
  ghcr.io/mxschll/harmonie:latest
```

Both paths are fixed inside the container:

| Container path | What it holds |
| --- | --- |
| `/music` | Your library, mounted read-only. Mount several under `/music/...` if you have more than one. |
| `/data` | `harmonie.db` and runtime state. This is the directory you back up or move. |

`MUSIC_DIR` is the host directory compose mounts at `/music`. The container's
`HARMONIE_LIBRARIES` is already `/music` and is not read from `.env`.

Configuration is otherwise the same `HARMONIE_*` environment set as a normal
install, so `-e HARMONIE_WORKERS=4` or a `.env` file works as documented in the
[README](README.md).

## Moving the database to another host

Scanning is CPU-bound; serving is not. Scan on the fastest machine available,
then move the database to the host that runs Jellyfin. Tracks are recognised by
content, so the library may sit at a different path on the second host.

On the fast host:

```bash
MUSIC_DIR=/path/to/music docker compose up
```

Wait for the scan to finish — `docker compose logs -f`, or
`docker compose exec harmonie harmonie status` — then stop it and copy the data
directory across:

```bash
docker compose down
rsync -a harmonie-data/ user@slow-host:/srv/harmonie/harmonie-data/
```

Start it the same way on the slow host, with `MUSIC_DIR` pointing at the library
there.

### Scan without serving

One pass, then exit:

```bash
MUSIC_DIR=/path/to/music docker compose run --rm harmonie scan
```

Any CLI subcommand works the same way: `status`, `info`, `similar`, `scans`.

## Running it as a service

`harmonie serve` runs the HTTP API and the scheduler: a scan on startup, then
one every `HARMONIE_SCAN_INTERVAL_HOURS` (24 by default). A health check calls
`/health` every 30 seconds.

The compose file sets `restart: unless-stopped`, `init: true` so signals reach
harmonie and exited workers are reaped, and `stop_grace_period: 60s` because a
scan in flight can take longer to wind down than Docker's default 10 seconds.
Killing it mid-scan is safe: unfinished tracks are analysed on the next run.

> [!IMPORTANT]
> **Analysis workers stay resident between scans.** The pool is built at the
> first scan and kept, each worker holding TensorFlow and the models in memory.
> An idle container with two workers uses 1.2 GB. The default is one worker per
> usable CPU, so set `HARMONIE_WORKERS` to what the host can afford in memory:
>
> ```
> HARMONIE_WORKERS=2
> ```

Disabling scanning means no pool is built at all:

```
HARMONIE_SCAN_ON_STARTUP=false
HARMONIE_SCAN_INTERVAL_HOURS=0
```

New files are not picked up while that is set. Scan on another host and copy the
data directory across.

## Switching from a native install

Stop the native harmonie, so its write-ahead log is folded into the database
file, then copy the database into the directory you mount at `/data`:

```bash
cp ~/.local/share/harmonie/harmonie.db ./harmonie-data/
```

On macOS the file lives in `~/Library/Application Support/harmonie/`.

The library is at `/music` in the container rather than its old path. The first
scan matches the existing analysis to it and reports `full=0`.

## CPU, architecture, and GPUs

The image is `linux/amd64` only and needs a CPU with AVX.

Analysis comes from `essentia-tensorflow`, which publishes wheels for x86-64
Linux and nothing else. There is no arm64 build, so ARM boards and NAS boxes can
only run this image under emulation, which is too slow to scan with. The bundled
TensorFlow is 2.5, built with AVX, so a CPU older than roughly 2011 fails at
import with `Illegal instruction`.

AVX2 is not required: TensorFlow uses it when the CPU has it and falls back when
it does not.

Check a host:

```bash
grep -o 'avx[2]*' /proc/cpuinfo | sort -u
docker run --rm ghcr.io/mxschll/harmonie:latest python -c "import essentia.standard; print('ok')"
```

Scanning is parallel. `HARMONIE_WORKERS` sets how many cores to spend on it; the
default uses every CPU the container may use, at roughly 1 GB of RAM each.

### GPUs

The bundled TensorFlow is a GPU-capable build — it looks for `libcuda.so.1` at
startup and logs the failure to load it — but the image ships no CUDA libraries,
so analysis runs on the CPU.

A GPU image would need a CUDA base matching TensorFlow 2.5 (CUDA 11.2 with cuDNN
8.1), run with `--gpus all` and the NVIDIA Container Toolkit. That is untested
here.

## Notes

The Essentia models are baked into the image, so a fresh container needs no
network access on first run.

Files written to `/data` are owned by root, since the container runs as root.
Add `user: "1000:1000"` in `compose.yaml`, and make the data directory writable
by that user, to change that.

Harmonie reports paths under `/music`, so revisit any path mappings configured
in the Jellyfin plugin. Tag-based matching is unaffected.
