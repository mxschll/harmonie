"""Configuration via environment variables (and optional .env file)."""

from __future__ import annotations

import logging
import logging.config
import os
import sys
from pathlib import Path
from typing import Annotated, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


def _default_data_dir() -> Path:
    """Default location for ``harmonie.db`` and runtime state.

    Resolves to the platform's user-data directory:

    * Linux: ``$XDG_DATA_HOME/harmonie`` (typically ``~/.local/share/harmonie``)
    * macOS: ``~/Library/Application Support/harmonie``
    * Windows: ``%LOCALAPPDATA%\\harmonie``

    Falls back to ``~/.local/share/harmonie`` if ``platformdirs`` is unavailable
    (it's a hard dep, but the fallback keeps tests robust).
    """
    try:
        from platformdirs import user_data_dir

        return Path(user_data_dir("harmonie", appauthor=False))
    except Exception:  # pragma: no cover - platformdirs is a hard dep
        return Path.home() / ".local" / "share" / "harmonie"


class Settings(BaseSettings):
    """Service configuration. Each field maps to a HARMONIE_* env var."""

    model_config = SettingsConfigDict(
        env_prefix="HARMONIE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Library / storage --------------------------------------------------
    libraries: Annotated[list[Path], NoDecode] = Field(
        default_factory=list,
        description="Absolute paths to scan for audio files.",
    )
    data_dir: Path = Field(
        default_factory=_default_data_dir,
        description=(
            "Where to store the SQLite database and runtime state. Defaults "
            "to the platform user-data directory (~/.local/share/harmonie on "
            "Linux, ~/Library/Application Support/harmonie on macOS)."
        ),
    )
    db_filename: str = Field(default="harmonie.db")

    # Analysis -----------------------------------------------------------
    backend: str = Field(default="effnet", description="effnet | musicextractor")
    workers: int = Field(default=0, description="0 = use CPU count.")

    # Scheduling ---------------------------------------------------------
    scan_interval_hours: float = Field(default=6.0, ge=0)
    scan_on_startup: bool = Field(default=True)

    # HTTP API -----------------------------------------------------------
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8842, ge=1, le=65535)
    api_key: Optional[str] = Field(default=None)
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)

    # Logging ------------------------------------------------------------
    log_level: str = Field(default="INFO")

    # ------------------------------------------------------------------

    @field_validator("libraries", mode="before")
    @classmethod
    def _parse_libraries(cls, v):
        if v is None or v == "":
            return []
        if isinstance(v, str):
            # Support comma- or colon-separated.
            sep = "," if "," in v else (":" if ":" in v else None)
            parts = [v] if sep is None else v.split(sep)
            return [Path(p.strip()) for p in parts if p.strip()]
        return v

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors(cls, v):
        if v is None or v == "":
            return []
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v

    @field_validator("backend")
    @classmethod
    def _normalize_backend(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in {"effnet", "musicextractor"}:
            raise ValueError(f"backend must be 'effnet' or 'musicextractor', got {v!r}")
        return v

    @field_validator("log_level")
    @classmethod
    def _normalize_log_level(cls, v: str) -> str:
        v = v.upper().strip()
        if v not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"invalid log level: {v!r}")
        return v

    # Convenience properties --------------------------------------------

    @property
    def db_path(self) -> Path:
        return self.data_dir / self.db_filename

    @property
    def worker_count(self) -> int:
        if self.workers > 0:
            return self.workers
        return max(1, os.cpu_count() or 1)


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Return the process-wide Settings instance, building it lazily."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def configure_logging(settings: Settings) -> None:
    """Set up root logging. Called once at process start."""
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "plain": {
                "format": "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                "datefmt": "%Y-%m-%dT%H:%M:%S",
            },
        },
        "handlers": {
            "default": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
                "formatter": "plain",
            },
        },
        "root": {"level": settings.log_level, "handlers": ["default"]},
        "loggers": {
            "tensorflow": {"level": "ERROR", "propagate": True},
            "uvicorn.access": {"level": "WARNING", "propagate": True},
        },
    }
    logging.config.dictConfig(config)
