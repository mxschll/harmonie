> [!NOTE]
> Everything below assumes an x86-64 host with AVX (Intel Sandy Bridge / AMD
> Bulldozer, 2011 or newer). See [CPU and architecture](#cpu-and-architecture).

## Run it

One command, once you point it at your music:

```bash
MUSIC_DIR=/path/to/music docker compose up -d
```

Harmonie listens on port 8842 and starts its first scan immediately. Watch it
with `docker compose logs -f`, or from the Jellyfin plugin's settings page.

Prefer plain Docker:

```bash
docker run -d --name harmonie \
  -p 8842:8842 \
  -v ./harmonie-data:/data \
  -v /path/to/music:/music:ro \
  ghcr.io/mxschll/harmonie:latest
```

Two paths matter, and both are fixed inside the container:

| Container path | What it holds |
| --- | --- |
| `/music` | Your library, mounted read-only. Mount several under `/music/...` if you have more than one. |
| `/data` | `harmonie.db` and runtime state. This is the directory you back up or move. |

Configuration is the same `HARMONIE_*` environment set as a normal install, so
`-e HARMONIE_WORKERS=4` or a `.env` file works as documented in the
[README](README.md). Leave `HARMONIE_LIBRARIES` and `HARMONIE_DATA_DIR` alone
unless you know why you are changing them.

## Moving the database to another host

Scanning is the expensive part: it decodes every track and runs the Essentia
models over it. Serving is cheap — similarity queries are arithmetic over
vectors already in the database. So scan on the fastest machine you have, then
move the result to whatever box runs Jellyfin.

This works because the container always sees the library at `/music`, so the
stored paths stay correct on a host whose library sits somewhere else. Copying
the library across is fine too: tracks are recognised by content, not location.

On the fast host:

```bash
MUSIC_DIR=/path/to/music docker compose up
```

Wait for the scan to finish — `docker compose logs -f`, or
`docker compose exec harmonie harmonie status`. Then stop it and copy the data
directory across:

```bash
docker compose down
rsync -a harmonie-data/ user@slow-host:/srv/harmonie/harmonie-data/
```

On the slow host, start it the same way, with `MUSIC_DIR` pointing at wherever
the library lives there.

Turn off scanning entirely if the second host should never analyse anything:

```bash
HARMONIE_SCAN_ON_STARTUP=false
HARMONIE_SCAN_INTERVAL_HOURS=0
```

New files added later will not be picked up while that is set. Either scan
again on the fast host and re-copy, or let the slow host scan on a schedule and
accept that it will take a while.

### Scan without serving

To run one pass and exit — useful in a cron job or on a machine you only borrow:

```bash
MUSIC_DIR=/path/to/music docker compose run --rm harmonie scan
```

Any CLI subcommand works the same way: `status`, `info`, `similar`, `scans`.

## Running it as a service

The container is a long-lived service, not a batch job. `harmonie serve` runs
the HTTP API and its own scheduler: a scan on startup, then one every
`HARMONIE_SCAN_INTERVAL_HOURS` (24 by default). Nothing external needs to
trigger it — no cron, no separate worker container.

The compose file sets `restart: unless-stopped`, `init: true` so signals reach
harmonie and dead workers get reaped, and a 60-second stop grace period, since
a scan in flight can take longer than Docker's default 10 seconds to wind down.
A hard kill is safe anyway: SQLite is crash-safe and unfinished tracks are
simply analysed again next time.

Docker's own health check calls `/health` every 30 seconds, so `docker ps` and
orchestrators can see whether the service is actually answering.

> [!IMPORTANT]
> **Analysis workers stay resident between scans.** The pool is built at the
> first scan and kept, so each worker holds TensorFlow and the models in memory
> for the life of the container — an idle container with two workers measured
> 1.2 GB. The default is one worker per core, so on a 12-core host you are
> holding that many copies permanently. Set `HARMONIE_WORKERS` to something
> your host can afford, especially on a machine that mostly serves:
>
> ```
> HARMONIE_WORKERS=2
> ```

For a host that should never analyse anything, disable scanning entirely and
the pool is never built:

```
HARMONIE_SCAN_ON_STARTUP=false
HARMONIE_SCAN_INTERVAL_HOURS=0
```

## Switching from a native install

Copy your existing database into the directory you mount at `/data` and start
the container as usual. Stop the native harmonie first, so its write-ahead log
is folded into the database file:

```bash
cp ~/.local/share/harmonie/harmonie.db ./harmonie-data/
```

On macOS the file lives in `~/Library/Application Support/harmonie/`.

The library now sits at `/music` instead of wherever it was, which the first
scan sorts out on its own: it should report `full=0` with everything skipped.

## CPU, architecture, and GPUs

The image is `linux/amd64` only, and needs a CPU with AVX.

Harmonie's analysis comes from `essentia-tensorflow`, which publishes wheels for
x86-64 Linux and nothing else — there is no arm64 build, so Raspberry Pis and
ARM NAS boxes cannot run this image except under emulation, which is far too
slow for scanning. The bundled TensorFlow is 2.5, built with AVX like every
official TensorFlow binary, so a CPU older than roughly 2011 fails at import
with `Illegal instruction`.

**AVX2 is not required.** TensorFlow uses it when the CPU has it and falls back
when it does not, so one image covers both. There is nothing to gain from a
second tag here: the only way to change the instruction baseline is compiling
TensorFlow and Essentia from source, which is a different project.

Check a host before you trust it with:

```bash
grep -o 'avx[2]*' /proc/cpuinfo | sort -u
docker run --rm ghcr.io/mxschll/harmonie:latest python -c "import essentia.standard; print('ok')"
```

Scanning is CPU-bound and parallel. Set `HARMONIE_WORKERS` to the number of
cores to spend on it; the default uses all of them, at roughly 1 GB of RAM each.

### GPUs

The TensorFlow bundled inside the wheel is a GPU-capable build — it looks for
`libcuda.so.1` at startup and logs a failure to load it — but this image ships
no CUDA libraries, so it always runs on the CPU.

Making the GPU usable would mean a second image built on a CUDA base matching
that TensorFlow (2.5's tested pairing is CUDA 11.2 with cuDNN 8.1), run with
`--gpus all` and the NVIDIA Container Toolkit. That is the one case where a
second tag would buy something. It is untested here, and worth measuring before
building: a scan also decodes every file on the CPU, so the model inference may
not be the part that is slow.

## Notes

The Essentia models are baked into the image, so a fresh container needs no
network access and no warm-up download.

Files written to `/data` are owned by root, since the container runs as root to
keep permissions simple. Add `user: "1000:1000"` in `compose.yaml` (and make the
data directory writable by that user) if that matters to you.

If you previously ran Harmonie outside Docker and configured path mappings in
the Jellyfin plugin, revisit them: Harmonie now reports paths under `/music`.
Tag-based matching is unaffected.
