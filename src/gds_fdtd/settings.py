"""
gds_fdtd simulation toolbox.

Package configuration: one pydantic-settings model, environment
prefix ``GDS_FDTD_``. Nothing here reads config files or the network — env
vars only, so behavior on a laptop, in CI, and inside a container is the
same mechanism.

    GDS_FDTD_CACHE_DIR=/scratch/cache
    GDS_FDTD_LOG_LEVEL=DEBUG
    GDS_FDTD_LOG_FORMAT=json     # JSON-lines logs for Modal/AWS aggregation

Telemetry is OFF by default and, even when enabled, only ever logs an
anonymous solver-name + duration line LOCALLY — there is no network
telemetry in 1.x; the setting exists so the policy is explicit.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GdsFdtdSettings(BaseSettings):
    """Runtime configuration, resolved from ``GDS_FDTD_*`` env vars."""

    model_config = SettingsConfigDict(env_prefix="GDS_FDTD_", extra="ignore")

    cache_dir: Path = Field(
        default_factory=lambda: Path.home() / ".cache" / "gds_fdtd",
        description="where run_cached() stores SMatrix results",
    )
    default_budget_fc: float | None = Field(
        None, ge=0, description="default Budget.max_flexcredits for jobs that set none"
    )
    log_level: str = "INFO"
    log_format: Literal["text", "json"] = "text"
    telemetry: bool = False  # local-only duration logging; never network


@lru_cache(maxsize=1)
def settings() -> GdsFdtdSettings:
    """The process-wide settings instance (env read once, cached)."""
    return GdsFdtdSettings()


def reset_settings() -> None:
    """Drop the cache so changed env vars are re-read (tests)."""
    settings.cache_clear()
