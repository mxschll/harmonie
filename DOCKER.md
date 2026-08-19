> [!NOTE]
> The image requires an x86-64 host with AVX (Intel Sandy Bridge / AMD
> Bulldozer, 2011 or newer). See
> [CPU, architecture, and GPUs](#cpu-architecture-and-gpus).

## Run it

```bash
docker run -d --name harmonie --init --restart unless-stopped \
  -p 127.0.0.1:8842:8842 \
  -v ./harmonie-data:/data \
  -v /path/to/music:/music:ro \
  ghcr.io/mxschll/harmonie:latest
```

Harmonie listens on port 8842 and starts its first scan immediately. Follow it
with `docker logs -f harmonie`, or from the Jellyfin plugin's settings page.

The port is published on `127.0.0.1` because the API takes no credentials unless
`HARMONIE_API_KEY` is set. Use `-p 0.0.0.0:8842:8842` to reach it from another
machine, such as a Jellyfin host on the LAN.

Both paths are fixed inside the container:

| Container path | What it holds |
| --- | --- |
| `/music` | Your library, mounted read-only. Mount several under `/music/...` if you have more than one. |
| `/data` | `harmonie.db` and runtime state. This is the directory you back up or move. |

With the `compose.yaml` from this repository:

```bash
MUSIC_DIR=/path/to/music docker compose up -d
```

`MUSIC_DIR` is the host directory compose mounts at `/music`; the container's
`HARMONIE_LIBRARIES` is already `/music` and is not read from `.env`. Compose
publishes on `127.0.0.1` as well, and `BIND_ADDRESS=0.0.0.0` opens it up.

Compose forwards `HARMONIE_WORKERS`, `HARMONIE_SCAN_INTERVAL_HOURS`,
`HARMONIE_SCAN_ON_STARTUP`, `HARMONIE_API_KEY`, `HARMONIE_CORS_ORIGINS` and
`HARMONIE_LOG_LEVEL` from the environment or a `.env` file. Add any other
`HARMONIE_*` setting from the [README](README.md) to the `environment:` block
yourself.

## Scan without serving

One pass, then exit:

```bash
docker run --rm \
  -v ./harmonie-data:/data \
  -v /path/to/music:/music:ro \
  ghcr.io/mxschll/harmonie:latest scan
```

With compose: `MUSIC_DIR=/path/to/music docker compose run --rm harmonie scan`.
Any CLI subcommand works the same way: `status`, `info`, `similar`, `scans`.

## Running it as a service

The container serves the HTTP API and scans on its own: once at startup, then
every `HARMONIE_SCAN_INTERVAL_HOURS` (24 by default). A health check calls
`/health` every 30 seconds.

`--init` reaps exited workers and lets signals reach harmonie; `--restart
unless-stopped` brings it back after a reboot. A scan in flight can take longer
to wind down than Docker's default 10 seconds, so stop it with `docker stop -t 60
harmonie`; the compose file sets that as `stop_grace_period`. Killing it mid-scan
is safe: unfinished tracks are analysed on the next run.

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

New files are not picked up while that is set.

## CPU, architecture, and GPUs

The image is `linux/amd64` only and needs a CPU with AVX.

Analysis comes from `essentia-tensorflow`, which publishes x86-64 Linux wheels
only. ARM boards and NAS boxes can run this image only under emulation, which is
too slow to scan with. The bundled TensorFlow is 2.5, built with AVX, so a CPU
older than roughly 2011 fails at import with `Illegal instruction`.

AVX2 is not required: TensorFlow uses it when the CPU has it and falls back when
it does not.

Check a host:

```bash
grep -o 'avx[2]*' /proc/cpuinfo | sort -u
docker run --rm ghcr.io/mxschll/harmonie:latest python -c "import essentia.standard; print('ok')"
```

Scanning is parallel. `HARMONIE_WORKERS` sets how many cores to spend on it; the
default uses every CPU the container may use, at roughly 1 GB of RAM each.

### GPUs (experimental)

The bundled TensorFlow has CUDA statically linked; its only external CUDA
dependency is the driver, `libcuda.so.1`. Install the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
and pass the card in:

```bash
docker run -d --name harmonie --gpus all \
  -p 8842:8842 \
  -v ./harmonie-data:/data \
  -v /path/to/music:/music:ro \
  ghcr.io/mxschll/harmonie:latest
```

In compose, add `gpus: all` to the service. A startup line reports what it found:

```
1 CUDA device(s) visible; running inference on GPU (experimental).
Workers share the card, so keep HARMONIE_WORKERS low
```

Only model inference moves to the GPU. Decoding, the mel spectrogram and the
musical descriptors are Essentia's CPU code and about three quarters of the work,
so expect at most 1.3x on a scan. Every worker opens its own TensorFlow session
on the same card, so raise `HARMONIE_WORKERS` with care.

Untested against real hardware. Set `CUDA_VISIBLE_DEVICES=` to force the CPU
path. Embedded GPU code covers `sm_35` through `sm_86` plus `compute_86` PTX, so
Ada and Hopper cards compile on first use.

## Notes

The Essentia models are baked into the image, so a fresh container needs no
network access on first run.

Files written to `/data` are owned by root, since the container runs as root.
Add `user: "1000:1000"` in `compose.yaml`, and make the data directory writable
by that user, to change that.

Harmonie reports paths under `/music`, so revisit any path mappings configured
in the Jellyfin plugin. Tag-based matching is unaffected.
