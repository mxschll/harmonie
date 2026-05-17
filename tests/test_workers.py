"""Tests for the worker-process initialization.

The WorkerPool itself spawns OS processes and loads TensorFlow, which makes
it impractical to test in isolation. These tests target ``_worker_init``
directly with stubbed dependencies, which is enough to verify the parts of
the contract callers care about: Essentia warnings get silenced for normal
log levels and stay on for DEBUG.
"""

from __future__ import annotations

import sys
import types

import pytest


@pytest.fixture
def stub_essentia(monkeypatch):
    """Inject a fake ``essentia`` module so ``_worker_init`` can flip its
    ``log.warningActive`` flag without us depending on real essentia."""
    fake = types.ModuleType("essentia")
    fake.log = types.SimpleNamespace(warningActive=True)
    monkeypatch.setitem(sys.modules, "essentia", fake)
    return fake


@pytest.fixture
def stub_extractor(monkeypatch):
    """Replace ``get_extractor`` so we don't try to load TF inside the test."""
    from harmonie import workers as workers_mod

    sentinel = object()
    monkeypatch.setattr(workers_mod, "get_extractor", lambda backend: sentinel)
    return sentinel


def test_worker_init_silences_essentia_warnings_at_info(
    stub_essentia, stub_extractor
):
    """Default log level — Essentia warnings get turned off."""
    from harmonie.workers import _worker_init

    _worker_init("effnet", "INFO")
    assert stub_essentia.log.warningActive is False


def test_worker_init_silences_essentia_warnings_at_warning(
    stub_essentia, stub_extractor
):
    """Any non-DEBUG level still silences."""
    from harmonie.workers import _worker_init

    _worker_init("effnet", "WARNING")
    assert stub_essentia.log.warningActive is False


def test_worker_init_keeps_warnings_at_debug(stub_essentia, stub_extractor):
    """DEBUG mode leaves Essentia warnings on for diagnostics."""
    from harmonie.workers import _worker_init

    _worker_init("effnet", "DEBUG")
    assert stub_essentia.log.warningActive is True


def test_worker_init_debug_case_insensitive(stub_essentia, stub_extractor):
    """The check is case-insensitive — settings.log_level is normalized to
    upper, but we shouldn't depend on that here."""
    from harmonie.workers import _worker_init

    _worker_init("effnet", "debug")
    assert stub_essentia.log.warningActive is True
