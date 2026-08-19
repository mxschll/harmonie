"""Worker count must reflect the CPUs a process may use, not the machine's.

``os.cpu_count()`` ignores container limits, so a container held to two CPUs on a
twelve-core host started twelve analysis workers, each holding its own copy of
the models in memory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harmonie import config as config_mod
from harmonie.config import Settings, available_cpus


@pytest.fixture
def no_cgroup(monkeypatch, tmp_path: Path):
    """Point the cgroup paths at files that do not exist."""
    for name in ("CGROUP_V2_CPU_MAX", "CGROUP_V1_QUOTA", "CGROUP_V1_PERIOD"):
        monkeypatch.setattr(config_mod, name, tmp_path / f"missing-{name}")


def _fake_affinity(monkeypatch, count: int):
    monkeypatch.setattr(
        config_mod.os,
        "sched_getaffinity",
        lambda _pid: set(range(count)),
        raising=False,
    )


def test_uses_affinity_rather_than_machine_size(monkeypatch, no_cgroup):
    """--cpuset-cpus and taskset restrict affinity while cpu_count stays put."""
    _fake_affinity(monkeypatch, 2)
    monkeypatch.setattr(config_mod.os, "cpu_count", lambda: 12)
    assert available_cpus() == 2


def test_cgroup_v2_quota_caps_the_count(monkeypatch, no_cgroup, tmp_path: Path):
    """`docker run --cpus=2` on a twelve-core host."""
    _fake_affinity(monkeypatch, 12)
    cpu_max = tmp_path / "cpu.max"
    cpu_max.write_text("200000 100000")
    monkeypatch.setattr(config_mod, "CGROUP_V2_CPU_MAX", cpu_max)
    assert available_cpus() == 2


def test_cgroup_v2_unlimited_is_ignored(monkeypatch, no_cgroup, tmp_path: Path):
    _fake_affinity(monkeypatch, 8)
    cpu_max = tmp_path / "cpu.max"
    cpu_max.write_text("max 100000")
    monkeypatch.setattr(config_mod, "CGROUP_V2_CPU_MAX", cpu_max)
    assert available_cpus() == 8


def test_fractional_quota_rounds_down(monkeypatch, no_cgroup, tmp_path: Path):
    """--cpus=1.5 gets one worker: a second would hold a whole model for half a
    core."""
    _fake_affinity(monkeypatch, 12)
    cpu_max = tmp_path / "cpu.max"
    cpu_max.write_text("150000 100000")
    monkeypatch.setattr(config_mod, "CGROUP_V2_CPU_MAX", cpu_max)
    assert available_cpus() == 1


def test_quota_below_one_cpu_still_gets_a_worker(
    monkeypatch, no_cgroup, tmp_path: Path
):
    _fake_affinity(monkeypatch, 4)
    cpu_max = tmp_path / "cpu.max"
    cpu_max.write_text("50000 100000")
    monkeypatch.setattr(config_mod, "CGROUP_V2_CPU_MAX", cpu_max)
    assert available_cpus() == 1


def test_cgroup_v1_quota_caps_the_count(monkeypatch, no_cgroup, tmp_path: Path):
    _fake_affinity(monkeypatch, 12)
    quota = tmp_path / "cpu.cfs_quota_us"
    period = tmp_path / "cpu.cfs_period_us"
    quota.write_text("300000")
    period.write_text("100000")
    monkeypatch.setattr(config_mod, "CGROUP_V1_QUOTA", quota)
    monkeypatch.setattr(config_mod, "CGROUP_V1_PERIOD", period)
    assert available_cpus() == 3


def test_cgroup_v1_unlimited_is_ignored(monkeypatch, no_cgroup, tmp_path: Path):
    _fake_affinity(monkeypatch, 6)
    quota = tmp_path / "cpu.cfs_quota_us"
    quota.write_text("-1")
    monkeypatch.setattr(config_mod, "CGROUP_V1_QUOTA", quota)
    assert available_cpus() == 6


def test_unreadable_cgroup_files_fall_back_to_affinity(
    monkeypatch, no_cgroup, tmp_path: Path
):
    _fake_affinity(monkeypatch, 5)
    garbage = tmp_path / "cpu.max"
    garbage.write_text("not a quota")
    monkeypatch.setattr(config_mod, "CGROUP_V2_CPU_MAX", garbage)
    assert available_cpus() == 5


def test_explicit_setting_wins(monkeypatch, no_cgroup, tmp_path: Path):
    _fake_affinity(monkeypatch, 2)
    settings = Settings(libraries=[], data_dir=tmp_path, workers=9)
    assert settings.worker_count == 9


def test_default_follows_available_cpus(monkeypatch, no_cgroup, tmp_path: Path):
    _fake_affinity(monkeypatch, 3)
    monkeypatch.setattr(config_mod.os, "cpu_count", lambda: 12)
    settings = Settings(libraries=[], data_dir=tmp_path)
    assert settings.workers == 0
    assert settings.worker_count == 3
