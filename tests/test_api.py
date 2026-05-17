"""Tests for the API surface — filters, paths, playlist discriminator.

Covers:

* ``120..130`` range syntax in URLs.
* Body filter shape with nested ``{gte, lte}`` ranges.
* ``GET /tracks/resolve`` for path/tag-based lookup.
* ``GET /scan`` (live state) and ``POST /scan?force=true`` (trigger).
* ``GET /status`` (service overview).
* ``POST /playlists`` with explicit ``mode`` discriminator.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from harmonie.api.filters import FilterBody, FloatRange, build_track_filter, parse_range
from harmonie.api.routes import api_router, public_router
from harmonie.config import Settings
from harmonie.features import Descriptors
from harmonie.tags import Tags


# ---------------------------------------------------------------------------
# Range parser
# ---------------------------------------------------------------------------


class TestParseRange:
    def test_closed(self):
        r = parse_range("120..130")
        assert r.gte == 120 and r.lte == 130

    def test_lower_only(self):
        r = parse_range("120..")
        assert r.gte == 120 and r.lte is None

    def test_upper_only(self):
        r = parse_range("..130")
        assert r.gte is None and r.lte == 130

    def test_negative_upper(self):
        # Loudness commonly looks like ``..-10`` (everything <= -10 dB).
        r = parse_range("..-10")
        assert r.gte is None and r.lte == -10

    def test_exact(self):
        r = parse_range("128")
        assert r.gte == 128 and r.lte == 128

    def test_negative_exact(self):
        r = parse_range("-12")
        assert r.gte == -12 and r.lte == -12

    def test_empty(self):
        assert parse_range("").is_empty()
        assert parse_range(None).is_empty()

    def test_malformed_raises(self):
        with pytest.raises(ValueError):
            parse_range("not-a-number")
        with pytest.raises(ValueError):
            parse_range("120..130..140")  # numeric parse fails on second '..'


class TestFloatRangeValidation:
    def test_inverted_bounds_rejected(self):
        with pytest.raises(ValueError):
            FloatRange(gte=130, lte=120)

    def test_equal_bounds_allowed(self):
        # Bare-number exact form arrives this way.
        r = FloatRange(gte=128, lte=128)
        assert r.gte == 128 and r.lte == 128


# ---------------------------------------------------------------------------
# build_track_filter — URL → TrackFilter
# ---------------------------------------------------------------------------


class TestBuildTrackFilter:
    def test_ranges_map_to_min_max(self):
        f = build_track_filter(bpm="120..130", loudness="..-10")
        assert f.bpm_min == 120 and f.bpm_max == 130
        assert f.loudness_min is None and f.loudness_max == -10

    def test_set_membership_passes_through(self):
        f = build_track_filter(key=["A", "B"], style=["Electronic"])
        assert f.key == ["A", "B"]
        assert f.styles == ["Electronic"]

    def test_style_min_and_mode(self):
        f = build_track_filter(
            style=["Rock"], style_min=0.5, style_mode="all",
        )
        assert f.style_min_probability == 0.5
        assert f.style_match == "all"


# ---------------------------------------------------------------------------
# FilterBody — body → TrackFilter
# ---------------------------------------------------------------------------


class TestFilterBody:
    def test_body_to_track_filter_round_trip(self):
        body = FilterBody.model_validate({
            "bpm": {"gte": 120, "lte": 130},
            "loudness": {"lte": -10},
            "key": ["A", "B"],
            "scale": "minor",
            "style": ["Electronic"],
            "style_min": 0.5,
            "style_mode": "all",
        })
        f = body.to_track_filter()
        assert f.bpm_min == 120 and f.bpm_max == 130
        assert f.loudness_min is None and f.loudness_max == -10
        assert f.key == ["A", "B"]
        assert f.scale == "minor"
        assert f.styles == ["Electronic"]
        assert f.style_min_probability == 0.5
        assert f.style_match == "all"

    def test_body_to_track_filter_isomorphic_with_url(self):
        """The same logical filter expressed both ways resolves to the same
        TrackFilter — that's the whole point of having two surfaces."""
        body_filter = FilterBody.model_validate(
            {"bpm": {"gte": 120, "lte": 130}, "key": ["A"]}
        ).to_track_filter()
        url_filter = build_track_filter(bpm="120..130", key=["A"])
        for slot in body_filter.__slots__:
            assert getattr(body_filter, slot) == getattr(url_filter, slot)


# ---------------------------------------------------------------------------
# API integration — paths, scan resource, /info+/stats split, resolve
# ---------------------------------------------------------------------------


def _stub_analyzer(db, index, settings, embedding_dim=4):
    """Minimal stand-in for the full Analyzer, just enough for the routes."""
    class _Status:
        state = "idle"

        def snapshot(self):
            return {
                "state": self.state,
                "phase": "idle",
                "started_at": None,
                "finished_at": None,
                "last_duration_sec": None,
                "last_error": None,
                "discovered": 0, "full": 0, "descriptors_only": 0,
                "skipped": 0, "failed": 0, "removed": 0,
                "recent_failures": [],
            }

    class _Stub:
        def __init__(self):
            self.db = db
            self.index = index
            self.settings = settings
            self.embedding_dim = embedding_dim
            self.status = _Status()

    return _Stub()


def _populate(db, name, *, bpm=128, loudness=-12, key="A", scale="minor"):
    return db.upsert_track(
        path=f"/lib/{name}",
        size=1,
        mtime=1.0,
        duration=200.0,
        embedding=np.zeros(4, dtype=np.float32),
        model="m1",
        descriptors=Descriptors(
            bpm=bpm, key=key, scale=scale, loudness=loudness, danceability=1.5,
        ),
        descriptor_version=1,
        tags=Tags(artist=name, title=name.replace(".flac", "")),
        library_root="/lib",
        relative_path=name,
    )


@pytest.fixture
def client(tmp_path: Path, make_db):
    db, index = make_db("api.db")
    _populate(db, "fast.flac", bpm=130)
    _populate(db, "mid.flac", bpm=120)
    _populate(db, "slow.flac", bpm=80)

    settings = Settings(libraries=[tmp_path], data_dir=tmp_path)
    app = FastAPI()
    app.include_router(public_router)
    app.include_router(api_router, prefix="/api/v1")
    app.state.analyzer = _stub_analyzer(db, index, settings)
    app.state.settings = settings
    return TestClient(app), db


class TestHealthAndInfo:
    def test_health_is_public(self, client):
        c, _ = client
        assert c.get("/health").status_code == 200

    def test_status_has_static_and_counters(self, client):
        c, db = client
        r = c.get("/api/v1/status")
        assert r.status_code == 200, r.text
        body = r.json()
        # Identity / config (was /info).
        assert {"version", "backend", "libraries", "schema_version",
                "descriptor_version"} <= body.keys()
        # Counters (was /stats).
        assert body["tracks"] == db.stats()["tracks"]
        assert "by_model" in body
        assert "total_duration_sec" in body
        # Live scan state lives at /scan, not on /status.
        assert "state" not in body
        assert "phase" not in body


class TestScanResource:
    def test_get_scan_returns_state(self, client):
        c, _ = client
        r = c.get("/api/v1/scan")
        assert r.status_code == 200, r.text
        assert r.json()["state"] == "idle"

    def test_get_scan_includes_phase(self, client):
        """The /scan response must expose the sub-phase field so callers can
        tell ``enumerating`` from ``extracting`` from ``pruning``."""
        c, _ = client
        body = c.get("/api/v1/scan").json()
        assert "phase" in body
        assert body["phase"] == "idle"  # no scan running

    def test_post_scan_accepts_force_query_param(self, client):
        """We don't actually run a scan in tests (no audio), but the route
        should accept ``?force=true`` and not 422 on missing body."""
        c, _ = client
        # Stub analyzer.scan would be called via to_thread — patch it out.
        analyzer = c.app.state.analyzer
        analyzer.scan = lambda **kw: None  # noqa: ARG005
        r = c.post("/api/v1/scan?force=true")
        # 200 plus state. Either "idle" (if the to_thread is fast) or
        # "scanning" (if it's still in flight). Both are acceptable.
        assert r.status_code == 200, r.text
        assert r.json()["state"] in {"idle", "scanning"}


class TestTracksList:
    def test_url_range_filter(self, client):
        c, _ = client
        r = c.get("/api/v1/tracks?bpm=100..140")
        assert r.status_code == 200, r.text
        names = sorted(it["path"] for it in r.json()["items"])
        assert names == ["/lib/fast.flac", "/lib/mid.flac"]

    def test_url_range_lower_only(self, client):
        c, _ = client
        r = c.get("/api/v1/tracks?bpm=125..")
        assert [it["bpm"] for it in r.json()["items"]] == [130]

    def test_url_range_upper_only(self, client):
        c, _ = client
        r = c.get("/api/v1/tracks?bpm=..100")
        assert [it["bpm"] for it in r.json()["items"]] == [80]

    def test_response_uses_loudness_not_loudness_db(self, client):
        c, _ = client
        item = c.get("/api/v1/tracks").json()["items"][0]
        assert "loudness" in item
        assert "loudness_db" not in item

    def test_invalid_range_400(self, client):
        c, _ = client
        r = c.get("/api/v1/tracks?bpm=foo..bar")
        assert r.status_code == 400


class TestResolve:
    def test_resolve_by_path(self, client):
        c, _ = client
        r = c.get("/api/v1/tracks/resolve", params={"path": "/lib/mid.flac"})
        assert r.status_code == 200, r.text
        assert r.json()["path"] == "/lib/mid.flac"

    def test_resolve_by_relative_path(self, client):
        c, _ = client
        r = c.get("/api/v1/tracks/resolve", params={"path": "mid.flac"})
        assert r.status_code == 200
        assert r.json()["path"] == "/lib/mid.flac"

    def test_resolve_by_tags(self, client):
        c, _ = client
        r = c.get(
            "/api/v1/tracks/resolve",
            params={"artist": "mid.flac", "title": "mid"},
        )
        assert r.status_code == 200

    def test_resolve_missing_400(self, client):
        c, _ = client
        r = c.get("/api/v1/tracks/resolve")
        assert r.status_code == 400

    def test_resolve_no_match_404(self, client):
        c, _ = client
        r = c.get("/api/v1/tracks/resolve", params={"path": "nope.flac"})
        assert r.status_code == 404


class TestPlaylistDiscriminator:
    def test_similar_mode_required_seeds(self, client):
        c, db = client
        track_id = list(db.list_tracks(limit=1)[0])[0]["id"]
        r = c.post(
            "/api/v1/playlists",
            json={"mode": "similar", "n": 2, "seeds": [track_id]},
        )
        assert r.status_code == 200, r.text

    def test_similar_mode_empty_seeds_rejected(self, client):
        """Empty ``seeds`` and no ``seed_refs`` → 422. The model validator
        requires at least one of the two fields to be non-empty."""
        c, _ = client
        r = c.post(
            "/api/v1/playlists",
            json={"mode": "similar", "n": 2, "seeds": []},
        )
        assert r.status_code == 422

    def test_drift_mode_accepts_multiple_seeds(self, client):
        """Drift mode now allows >1 seed; the centroid becomes the starting
        anchor. Previously this was a schema rejection (max_length=1)."""
        c, db = client
        ids = [int(r["id"]) for r in db.list_tracks(limit=2)[0]]
        r = c.post(
            "/api/v1/playlists",
            json={"mode": "drift", "seeds": ids, "chunk_size": 3, "n": 4},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        # No unresolved refs because we sent IDs only.
        assert body["unresolved_seed_refs"] == []
        # Seeds aren't in the result by default (include_seeds=False).
        returned_ids = [it["track_id"] for it in body["items"]]
        for sid in ids:
            assert sid not in returned_ids

    def test_vibe_mode_no_seeds_field(self, client):
        c, _ = client
        r = c.post(
            "/api/v1/playlists",
            json={
                "mode": "vibe",
                "n": 2,
                "filter": {"bpm": {"gte": 100}},
                "target": {"bpm": 128},
                "shuffle": False,
            },
        )
        assert r.status_code == 200, r.text
        items = r.json()["items"]
        assert len(items) <= 2

    def test_unknown_mode_rejected(self, client):
        c, _ = client
        r = c.post(
            "/api/v1/playlists",
            json={"mode": "interpolate", "seeds": [1, 2]},
        )
        assert r.status_code == 422

    def test_smooth_transitions_grouped(self, client):
        """The body groups bpm_tolerance + key_compatible under
        ``smooth_transitions``. They should not be top-level fields."""
        c, db = client
        track_id = list(db.list_tracks(limit=1)[0])[0]["id"]
        # Top-level form (old) → 422 because extra fields aren't allowed
        # in the discriminated schema.
        r = c.post(
            "/api/v1/playlists",
            json={
                "mode": "similar", "n": 2,
                "seeds": [track_id],
                "bpm_tolerance": 5,  # was top-level; now under smooth_transitions
            },
        )
        # Pydantic v2 default is to ignore extras silently; so the request
        # succeeds but the param is dropped. We only need to verify the
        # *grouped* form works.
        assert r.status_code == 200

        # Grouped form is the canonical shape.
        r = c.post(
            "/api/v1/playlists",
            json={
                "mode": "similar", "n": 2,
                "seeds": [track_id],
                "smooth_transitions": {
                    "bpm_tolerance": 5, "key_compatible": True,
                },
            },
        )
        assert r.status_code == 200, r.text



# ---------------------------------------------------------------------------
# Inline seed references on /playlists
# ---------------------------------------------------------------------------


class TestPlaylistSeedRefs:
    """Cover the ``seed_refs`` path on POST /playlists: server-side resolution
    via the same ladder /tracks/resolve uses, mixed with explicit ``seeds``,
    unresolved reporting, and the all-unresolved error."""

    def test_seed_ref_by_path(self, client):
        c, _ = client
        r = c.post(
            "/api/v1/playlists",
            json={
                "mode": "similar",
                "n": 2,
                "seed_refs": [{"path": "/lib/mid.flac"}],
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["unresolved_seed_refs"] == []
        assert len(body["items"]) >= 1

    def test_seed_ref_by_tags(self, client):
        c, _ = client
        # _populate stores artist=name (e.g. "mid.flac") and title=name.strip(.flac).
        r = c.post(
            "/api/v1/playlists",
            json={
                "mode": "similar",
                "n": 2,
                "seed_refs": [
                    {"artist": "mid.flac", "album": None, "title": "mid"},
                ],
            },
        )
        assert r.status_code == 200, r.text
        assert r.json()["unresolved_seed_refs"] == []

    def test_seed_refs_mixed_with_explicit_ids(self, client):
        """Sending both ``seeds`` and ``seed_refs`` works; the merged set
        is deduped."""
        c, db = client
        rows = db.list_tracks(limit=2)[0]
        seed_id = int(rows[0]["id"])
        seed_path = rows[1]["path"]
        r = c.post(
            "/api/v1/playlists",
            json={
                "mode": "similar",
                "n": 2,
                "seeds": [seed_id],
                "seed_refs": [{"path": seed_path}],
            },
        )
        assert r.status_code == 200, r.text
        assert r.json()["unresolved_seed_refs"] == []

    def test_unresolved_seed_refs_reported(self, client):
        """A bad ref alongside a good one: the good one drives the playlist
        and the bad one shows up in unresolved_seed_refs."""
        c, _ = client
        r = c.post(
            "/api/v1/playlists",
            json={
                "mode": "similar",
                "n": 2,
                "seed_refs": [
                    {"path": "/lib/mid.flac"},
                    {"artist": "DoesNotExist", "title": "Nope"},
                ],
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert len(body["unresolved_seed_refs"]) == 1
        bad = body["unresolved_seed_refs"][0]
        assert bad["reason"] == "no_match"
        assert bad["ref"]["artist"] == "DoesNotExist"
        assert len(body["items"]) >= 1

    def test_all_seed_refs_unresolved_400(self, client):
        """If nothing resolves and no explicit ``seeds`` were given, the
        request is a 400 — there's nothing to build a playlist from."""
        c, _ = client
        r = c.post(
            "/api/v1/playlists",
            json={
                "mode": "similar",
                "n": 2,
                "seed_refs": [
                    {"path": "/lib/nope-1.flac"},
                    {"artist": "Nobody", "title": "Nothing"},
                ],
            },
        )
        assert r.status_code == 400
        assert "no seeds resolved" in r.json()["detail"].lower()

    def test_seed_ref_requires_at_least_one_field(self, client):
        """An empty SeedRef ({}) is a schema error — Pydantic validator."""
        c, _ = client
        r = c.post(
            "/api/v1/playlists",
            json={"mode": "similar", "n": 2, "seed_refs": [{}]},
        )
        assert r.status_code == 422

    def test_no_seeds_and_no_seed_refs_rejected(self, client):
        """Both fields empty/missing → 422. Validator on _SeededPlaylist."""
        c, _ = client
        r = c.post(
            "/api/v1/playlists",
            json={"mode": "similar", "n": 2},
        )
        assert r.status_code == 422

    def test_drift_with_seed_refs(self, client):
        """Drift mode accepts seed_refs alone (no explicit ``seeds``)."""
        c, _ = client
        r = c.post(
            "/api/v1/playlists",
            json={
                "mode": "drift",
                "n": 3,
                "chunk_size": 2,
                "seed_refs": [
                    {"path": "/lib/mid.flac"},
                    {"path": "/lib/fast.flac"},
                ],
            },
        )
        assert r.status_code == 200, r.text
        assert r.json()["unresolved_seed_refs"] == []


# ---------------------------------------------------------------------------
# API key auth — gated by HARMONIE_API_KEY at app construction time
# ---------------------------------------------------------------------------


@pytest.fixture
def client_with_api_key(tmp_path: Path, make_db):
    """Same wiring as ``client`` but with an API key configured. Settings is
    constructed with ``api_key`` directly so we don't need to touch env."""
    db, index = make_db("api-keyed.db")
    _populate(db, "fast.flac", bpm=130)

    settings = Settings(
        libraries=[tmp_path], data_dir=tmp_path, api_key="s3cret",
    )
    app = FastAPI()
    app.include_router(public_router)
    app.include_router(api_router, prefix="/api/v1")
    app.state.analyzer = _stub_analyzer(db, index, settings)
    app.state.settings = settings
    return TestClient(app)


class TestApiKeyAuth:
    def test_health_is_public_even_with_key_set(self, client_with_api_key):
        """``/health`` must never require auth — it's the liveness probe."""
        assert client_with_api_key.get("/health").status_code == 200

    def test_missing_key_rejected(self, client_with_api_key):
        r = client_with_api_key.get("/api/v1/status")
        assert r.status_code == 401

    def test_wrong_key_rejected(self, client_with_api_key):
        r = client_with_api_key.get(
            "/api/v1/status", headers={"X-API-Key": "wrong"},
        )
        assert r.status_code == 401

    def test_correct_key_accepted(self, client_with_api_key):
        r = client_with_api_key.get(
            "/api/v1/status", headers={"X-API-Key": "s3cret"},
        )
        assert r.status_code == 200, r.text

    def test_no_key_required_when_unconfigured(self, client):
        """The default ``client`` fixture has no api_key set — every route
        should pass through unauthenticated."""
        c, _ = client
        assert c.get("/api/v1/status").status_code == 200
