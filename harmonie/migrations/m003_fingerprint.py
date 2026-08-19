"""Content fingerprints.

Adds ``tracks.fingerprint``: a hash of the file's size plus its first and last
64 KiB. Analysis is keyed on the absolute path, with size and mtime deciding
whether a file changed, and mtime does not survive being copied between systems.
A fingerprint identifies the same audio wherever it turns up, so a library that
moved or was copied is matched to its existing rows instead of analysed again.

Existing rows are backfilled during scans; the column stays nullable so an
un-backfilled row simply falls back to matching on library-relative path.
"""

from __future__ import annotations

import sqlite3

_STATEMENTS = [
    "ALTER TABLE tracks ADD COLUMN fingerprint TEXT",
    "CREATE INDEX idx_tracks_fingerprint ON tracks(fingerprint)",
]


def upgrade(conn: sqlite3.Connection) -> None:
    for stmt in _STATEMENTS:
        conn.execute(stmt)
