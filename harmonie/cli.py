"""harmonie CLI. Thin wrapper over the same modules used by the service.

Subcommands:

* ``serve``        — run the HTTP service (uvicorn + scheduler)
* ``scan``         — run one analysis pass and exit
* ``info <id|path>`` — print one track's stored info
* ``similar <id>``  — top-N similar to a track id
* ``list``          — list tracks (optional filters)
* ``status``        — print db stats and last scan status
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .config import configure_logging, get_settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt_duration(seconds: float) -> str:
    s = int(round(seconds))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def _fmt_opt(val, spec: str = "g", missing: str = "—") -> str:
    if val is None:
        return missing
    if isinstance(val, str):
        return val
    try:
        return format(val, spec)
    except (TypeError, ValueError):
        return str(val)


def _open_resources():
    """Open the DB plus a fresh EmbeddingIndex. Returned together because
    every read-side command in this CLI needs both."""
    settings = get_settings()
    configure_logging(settings)
    from .db import Database
    from .index import EmbeddingIndex

    db = Database(settings.db_path)
    index = EmbeddingIndex(db)
    return settings, db, index


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_migrate(args: argparse.Namespace) -> int:
    """Apply pending schema migrations and exit. Useful as a separate
    deploy step before bringing up `serve` or `scan`."""
    settings = get_settings()
    configure_logging(settings)
    import sqlite3

    from .db import Database
    from .migrations import (
        CURRENT_SCHEMA_VERSION,
        MigrationError,
        get_schema_version,
    )

    # Ensure the data directory exists before we touch SQLite — sqlite3
    # won't create it for us, but Database() would.
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)

    pre_conn = sqlite3.connect(settings.db_path)
    try:
        before = get_schema_version(pre_conn)
    finally:
        pre_conn.close()

    try:
        db = Database(settings.db_path)
    except MigrationError as e:
        print(f"migrate: {e}", file=sys.stderr)
        return 1
    try:
        after = get_schema_version(db._conn)
    finally:
        db.close()

    if before == after:
        print(f"Already at schema version {after}. Nothing to do.")
    else:
        print(f"Migrated database from version {before} to {after}.")
    if after != CURRENT_SCHEMA_VERSION:  # pragma: no cover - defensive
        print(
            f"Warning: latest known version is {CURRENT_SCHEMA_VERSION}, "
            f"DB ended up at {after}.",
            file=sys.stderr,
        )
        return 2
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Run the FastAPI service via uvicorn."""
    import uvicorn

    settings = get_settings()
    configure_logging(settings)
    # uvicorn wants an import string when reload=True; we don't reload in
    # production, so pass the app directly.
    from .api.app import create_app

    app = create_app(settings)
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_config=None,  # we configured logging ourselves
        access_log=False,
    )
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    settings = get_settings()
    configure_logging(settings)
    from .analyzer import Analyzer

    analyzer = Analyzer(settings)
    try:
        analyzer.scan(force=args.force)
        snap = analyzer.status.snapshot()
        if args.json:
            print(json.dumps(snap, indent=2))
        else:
            print(
                f"Discovered: {snap['discovered']}   "
                f"Full: {snap['full']}   "
                f"Descriptors-only: {snap['descriptors_only']}   "
                f"Skipped: {snap['skipped']}   "
                f"Failed: {snap['failed']}   "
                f"Removed: {snap['removed']}"
            )
        return 0 if snap["failed"] == 0 else 2
    finally:
        analyzer.stop()


def cmd_info(args: argparse.Namespace) -> int:
    settings = get_settings()
    configure_logging(settings)
    from .db import Database

    db = Database(settings.db_path)
    try:
        if args.target.isdigit():
            row = db.get_track_by_id(int(args.target))
        else:
            # Try the path the user typed first, then fall back to a
            # resolved version. Paths get stored at scan time as the
            # walker saw them, which may not be the canonical filesystem
            # path — especially with symlinked library mounts (``/data/
            # music`` → ``/mnt/music``) or with directories like ``/lib``
            # that are themselves symlinks on most Linux distros.
            row = db.get_track_by_path(args.target)
            if row is None:
                resolved = str(Path(args.target).expanduser().resolve())
                if resolved != args.target:
                    row = db.get_track_by_path(resolved)
        if row is None:
            print(f"Not in database: {args.target}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(row, indent=2, default=str))
            return 0
        key_disp = (
            f"{row['key']} {row['scale']}"
            if row.get("key") and row.get("scale")
            else _fmt_opt(row.get("key"))
        )
        print(f"ID:             {row['id']}")
        print(f"Path:           {row['path']}")
        if row.get("library_root"):
            print(f"Library:        {row['library_root']}")
            print(f"Relative path:  {row['relative_path']}")
        print(f"Duration:       {_fmt_duration(row['duration'])}")
        print(f"Model:          {row['model']}  (dim={row['embedding_dim']})")
        print(f"Descriptor v:   {row['descriptor_version']}")
        # Tags
        if row.get("title") or row.get("artist") or row.get("album"):
            print("Tags:")
            print(f"  Artist:       {_fmt_opt(row.get('artist'))}")
            print(f"  Album:        {_fmt_opt(row.get('album'))}")
            print(f"  Title:        {_fmt_opt(row.get('title'))}")
            print(f"  Track #:      {_fmt_opt(row.get('track_number'))}")
        print(f"BPM:            {_fmt_opt(row['bpm'], '.1f')}"
              f"   confidence: {_fmt_opt(row.get('bpm_confidence'), '.2f')}")
        print(f"Key:            {key_disp}"
              f"   strength: {_fmt_opt(row.get('key_strength'), '.2f')}")
        print(f"Loudness (RG):  {_fmt_opt(row['loudness'], '.2f')} dB")
        print(f"Danceability:   {_fmt_opt(row['danceability'], '.2f')}")
        print(f"Onset rate:     {_fmt_opt(row['onset_rate'], '.2f')}/s")
        return 0
    finally:
        db.close()


def cmd_similar(args: argparse.Namespace) -> int:
    from .similarity import find_similar_to_id

    _settings, db, index = _open_resources()
    try:
        try:
            matches = find_similar_to_id(db, index, int(args.track_id), n=args.n)
        except KeyError as e:
            print(str(e), file=sys.stderr)
            return 1
        if args.json:
            print(
                json.dumps(
                    [{"track_id": m.track_id, "path": m.path, "score": m.score}
                     for m in matches],
                    indent=2,
                )
            )
            return 0
        for i, m in enumerate(matches, 1):
            print(f"{i:>3}. {m.score:.4f}  [{m.track_id}] {m.path}")
        return 0
    finally:
        db.close()


def cmd_list(args: argparse.Namespace) -> int:
    settings = get_settings()
    configure_logging(settings)
    from .api.filters import build_track_filter
    from .db import Database

    db = Database(settings.db_path)
    try:
        try:
            f = build_track_filter(
                bpm=args.bpm,
                danceability=args.danceability,
                loudness=args.loudness,
                key=[args.key] if args.key else None,
                scale=args.scale,
            )
        except ValueError as e:
            print(f"list: invalid range filter: {e}", file=sys.stderr)
            return 1
        rows, total = db.list_tracks(filter=f, limit=args.limit, offset=args.offset)
        if args.json:
            print(json.dumps({"items": rows, "total": total}, indent=2, default=str))
            return 0
        print(
            f"{'id':>5}  {'len':>7}  {'bpm':>5}  {'key':>5}  {'dance':>5}  "
            f"{'loud':>6}  path"
        )
        for r in rows:
            key_str = (
                f"{r['key']}{'m' if (r.get('scale') == 'minor') else ''}"
                if r.get("key") else "—"
            )
            print(
                f"{r['id']:>5}  "
                f"{_fmt_duration(r['duration']):>7}  "
                f"{_fmt_opt(r['bpm'], '.1f'):>5}  "
                f"{key_str:>5}  "
                f"{_fmt_opt(r['danceability'], '.2f'):>5}  "
                f"{_fmt_opt(r['loudness'], '.1f'):>6}  "
                f"{r['path']}"
            )
        print(f"\n{len(rows)} of {total} track(s).")
        return 0
    finally:
        db.close()


def cmd_status(args: argparse.Namespace) -> int:
    settings = get_settings()
    configure_logging(settings)
    from .db import Database

    db = Database(settings.db_path)
    try:
        s = db.stats()
    finally:
        db.close()
    if args.json:
        print(json.dumps(s, indent=2, default=str))
        return 0
    total_min = s["total_duration_sec"] / 60.0
    db_mb = s["db_size_bytes"] / (1024 * 1024)
    print(f"Database:     {s['db_path']}")
    print(f"Tracks:       {s['tracks']}")
    print(f"Total audio:  {total_min:.1f} min")
    print(f"DB size:      {db_mb:.2f} MiB")
    if s["by_model"]:
        print("By model:")
        for model, count in s["by_model"].items():
            print(f"  {model}: {count}")
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="harmonie",
        description="Audio similarity service. CLI for ops + service launch.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    psv = sub.add_parser("serve", help="Run the HTTP service.")
    psv.set_defaults(func=cmd_serve)

    pmig = sub.add_parser(
        "migrate",
        help="Apply any pending DB schema migrations and exit.",
    )
    pmig.set_defaults(func=cmd_migrate)

    psc = sub.add_parser("scan", help="Run one analysis pass and exit.")
    psc.add_argument("--force", action="store_true", help="Re-extract everything.")
    psc.add_argument("--json", action="store_true")
    psc.set_defaults(func=cmd_scan)

    pi = sub.add_parser("info", help="Show stored info for one track.")
    pi.add_argument("target", help="Track ID or path.")
    pi.add_argument("--json", action="store_true")
    pi.set_defaults(func=cmd_info)

    psi = sub.add_parser("similar", help="Top-N similar tracks for a given track ID.")
    psi.add_argument("track_id", help="Track ID.")
    psi.add_argument("-n", type=int, default=10)
    psi.add_argument("--json", action="store_true")
    psi.set_defaults(func=cmd_similar)

    pl = sub.add_parser("list", help="List tracks with optional filters.")
    pl.add_argument(
        "--bpm",
        help="BPM range (e.g. ``120..130``, ``120..``, ``..130``, or ``128``).",
    )
    pl.add_argument(
        "--danceability",
        help="Danceability range, same syntax as ``--bpm``.",
    )
    pl.add_argument(
        "--loudness",
        help="Loudness range in dB, same syntax (e.g. ``..-10``).",
    )
    pl.add_argument("--key")
    pl.add_argument("--scale", choices=["major", "minor"])
    pl.add_argument("--limit", type=int, default=100)
    pl.add_argument("--offset", type=int, default=0)
    pl.add_argument("--json", action="store_true")
    pl.set_defaults(func=cmd_list)

    pst = sub.add_parser("status", help="Show database stats.")
    pst.add_argument("--json", action="store_true")
    pst.set_defaults(func=cmd_status)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
