"""Tests for ``harmonie.config.Settings``.

Mostly verifies that the defaults are sensible and that the platform-aware
``data_dir`` default lands somewhere writable instead of polluting the user's
working directory.
"""

from __future__ import annotations

from pathlib import Path

from harmonie.config import Settings, _default_data_dir


def test_default_data_dir_is_absolute_writable_path():
    """The default must not be a relative ``./data`` style path. That would
    create a directory wherever the user happened to run ``harmonie serve``,
    which is the bug this default is meant to avoid."""
    p = _default_data_dir()
    assert isinstance(p, Path)
    assert p.is_absolute(), f"expected absolute path, got {p!r}"
    # Should look like a user-data dir, not the current dir.
    s = str(p)
    assert "harmonie" in s.lower()


def test_settings_uses_default_data_dir(monkeypatch):
    """Settings without HARMONIE_DATA_DIR set should pick up the platform
    default and therefore be safe to instantiate from any cwd."""
    monkeypatch.delenv("HARMONIE_DATA_DIR", raising=False)
    s = Settings()
    assert s.data_dir == _default_data_dir()
    assert s.db_path == s.data_dir / "harmonie.db"


def test_settings_respects_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("HARMONIE_DATA_DIR", str(tmp_path))
    s = Settings()
    assert s.data_dir == tmp_path
    assert s.db_path == tmp_path / "harmonie.db"


def test_settings_explicit_constructor_arg_wins(tmp_path, monkeypatch):
    """Constructor args (used by tests) take precedence over env."""
    monkeypatch.setenv("HARMONIE_DATA_DIR", "/tmp/should-not-be-used")
    s = Settings(data_dir=tmp_path)
    assert s.data_dir == tmp_path


def test_job_timeout_zero_disables(monkeypatch):
    from harmonie.config import Settings

    assert Settings(job_timeout_sec=0).job_timeout is None
    assert Settings(job_timeout_sec=45).job_timeout == 45.0
    assert Settings().job_timeout == 300.0

    monkeypatch.setenv("HARMONIE_JOB_TIMEOUT_SEC", "0")
    assert Settings().job_timeout is None


def test_job_timeout_rejects_negative():
    import pytest

    from harmonie.config import Settings

    with pytest.raises(ValueError):
        Settings(job_timeout_sec=-1)
