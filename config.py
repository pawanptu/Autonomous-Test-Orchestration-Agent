"""Central configuration for the Autonomous Test Orchestration Agent.

Everything tunable lives here: feature flags, thresholds, model routing,
timeouts and filesystem layout. Nothing in this module reads or holds a secret
value other than the API key, which is pulled from the environment and never
echoed back (see :meth:`Settings.safe_dict`).

Design note
-----------
We deliberately avoid ``pydantic-settings``. A frozen dataclass built by an
explicit ``from_env`` classmethod keeps the dependency surface small, makes
every coercion visible, and lets unit tests construct a ``Settings`` object
without mutating the process environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final

from dotenv import load_dotenv

# Load a local .env if present. Values already in the real environment win.
load_dotenv(override=False)

# --------------------------------------------------------------------------
# Filesystem layout
# --------------------------------------------------------------------------
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
REPORTS_DIR: Final[Path] = PROJECT_ROOT / "reports"
RUNS_DIR: Final[Path] = REPORTS_DIR / "runs"
BASELINES_DIR: Final[Path] = REPORTS_DIR / "baselines"
GENERATED_TESTS_DIRNAME: Final[str] = "generated_tests"

# --------------------------------------------------------------------------
# Model routing.
#
# Rate-limit awareness is a hard requirement: the 70B model is reserved for
# judgment (coverage evaluation, risk ranking, defect classification,
# confidence scoring, orchestrator routing) and the small model does the
# mechanical plan-to-code translation. Never burn 70B on codegen.
# --------------------------------------------------------------------------
MODEL_REASONING: Final[str] = "llama-3.3-70b-versatile"
MODEL_CODEGEN: Final[str] = "llama-3.1-8b-instant"
MODEL_CODEGEN_ALT: Final[str] = "openai/gpt-oss-20b"

GROQ_BASE_URL: Final[str] = "https://api.groq.com/openai/v1"

# --------------------------------------------------------------------------
# Hard orchestration constants required by the specification.
# --------------------------------------------------------------------------
REPLAN_CAP: Final[int] = 2
"""Maximum number of Planner revisions the coverage gate may demand.

After this many failed gates the orchestrator force-proceeds and records
``force_proceeded=True`` plus an explicit limitation in the final report.
"""

CONFIDENCE_AUTO_APPLY_THRESHOLD: Final[float] = 0.6
"""Healer confidence at or above which a script fix may be auto-applied.

Strictly below this value the patch is *not* applied; the finding is queued
for human review with its evidence. This is a different code branch, not a
different log level.
"""

HEAL_RERUN_CAP: Final[int] = 1
"""A healed test is re-run at most once, to bound pipeline duration."""


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    return default if raw is None or raw.strip() == "" else raw.strip()


@dataclass(frozen=True)
class Settings:
    """Immutable runtime configuration.

    Use :meth:`from_env` in application code; construct directly in tests so
    that no environment mutation is required.
    """

    # ---- LLM -------------------------------------------------------------
    groq_api_key: str = ""
    groq_base_url: str = GROQ_BASE_URL
    model_reasoning: str = MODEL_REASONING
    model_codegen: str = MODEL_CODEGEN
    llm_timeout_s: float = 90.0
    llm_max_retries: int = 5
    llm_backoff_base_s: float = 1.5
    llm_backoff_max_s: float = 45.0
    llm_temperature_reasoning: float = 0.1
    llm_temperature_codegen: float = 0.2
    llm_max_tokens: int = 4096
    llm_offline_mode: bool = False
    """When true no network call is made and a clearly-labelled deterministic
    stub provider answers every prompt. This exists so the graph, API and UI
    can be smoke-tested without an API key. Any run made in this mode is
    stamped ``llm_provider = "offline-stub"`` in the report and carries a loud
    limitation entry. It is NOT the real agent."""

    # ---- Crawl -----------------------------------------------------------
    crawl_max_pages: int = 12
    crawl_max_depth: int = 2
    crawl_page_timeout_ms: int = 20_000
    crawl_settle_ms: int = 700
    crawl_same_origin_only: bool = True

    # ---- Browser / execution --------------------------------------------
    headless: bool = True
    viewport_width: int = 1280
    viewport_height: int = 900
    nav_timeout_ms: int = 25_000
    action_timeout_ms: int = 8_000
    test_timeout_s: float = 90.0
    max_parallel_flows: int = 3
    slow_mo_ms: int = 0

    # ---- Visual diff -----------------------------------------------------
    visual_diff_threshold: float = 0.02
    """Fraction of changed pixels above which a visual regression is reported."""
    visual_pixel_tolerance: int = 24
    """Per-channel 0-255 delta below which two pixels count as identical."""

    # ---- Generation ------------------------------------------------------
    max_flows_to_generate: int = 12
    selector_validation_candidates: int = 4

    # ---- Feature flags ---------------------------------------------------
    enable_prd_gap_analysis: bool = False
    enable_intent_bias: bool = True
    enable_parallel_execution: bool = False
    enable_regression_radar: bool = True
    enable_visual_diff: bool = True

    # ---- Misc ------------------------------------------------------------
    default_target_url: str = "https://books.toscrape.com/"
    log_level: str = "INFO"
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_base_url: str = "http://127.0.0.1:8000"
    max_concurrent_runs: int = 2

    # ---------------------------------------------------------------------
    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from the process environment / .env file."""
        return cls(
            groq_api_key=_env_str("GROQ_API_KEY", ""),
            groq_base_url=_env_str("GROQ_BASE_URL", GROQ_BASE_URL),
            model_reasoning=_env_str("MODEL_REASONING", MODEL_REASONING),
            model_codegen=_env_str("MODEL_CODEGEN", MODEL_CODEGEN),
            llm_timeout_s=_env_float("LLM_TIMEOUT_S", 90.0),
            llm_max_retries=_env_int("LLM_MAX_RETRIES", 5),
            llm_backoff_base_s=_env_float("LLM_BACKOFF_BASE_S", 1.5),
            llm_backoff_max_s=_env_float("LLM_BACKOFF_MAX_S", 45.0),
            llm_max_tokens=_env_int("LLM_MAX_TOKENS", 4096),
            llm_offline_mode=_env_bool("LLM_OFFLINE_MODE", False),
            crawl_max_pages=_env_int("CRAWL_MAX_PAGES", 12),
            crawl_max_depth=_env_int("CRAWL_MAX_DEPTH", 2),
            crawl_page_timeout_ms=_env_int("CRAWL_PAGE_TIMEOUT_MS", 20_000),
            headless=_env_bool("HEADLESS", True),
            viewport_width=_env_int("VIEWPORT_WIDTH", 1280),
            viewport_height=_env_int("VIEWPORT_HEIGHT", 900),
            nav_timeout_ms=_env_int("NAV_TIMEOUT_MS", 25_000),
            action_timeout_ms=_env_int("ACTION_TIMEOUT_MS", 8_000),
            test_timeout_s=_env_float("TEST_TIMEOUT_S", 90.0),
            max_parallel_flows=_env_int("MAX_PARALLEL_FLOWS", 3),
            slow_mo_ms=_env_int("SLOW_MO_MS", 0),
            visual_diff_threshold=_env_float("VISUAL_DIFF_THRESHOLD", 0.02),
            visual_pixel_tolerance=_env_int("VISUAL_PIXEL_TOLERANCE", 24),
            max_flows_to_generate=_env_int("MAX_FLOWS_TO_GENERATE", 12),
            enable_prd_gap_analysis=_env_bool("ENABLE_PRD_GAP_ANALYSIS", False),
            enable_intent_bias=_env_bool("ENABLE_INTENT_BIAS", True),
            enable_parallel_execution=_env_bool("ENABLE_PARALLEL_EXECUTION", False),
            enable_regression_radar=_env_bool("ENABLE_REGRESSION_RADAR", True),
            enable_visual_diff=_env_bool("ENABLE_VISUAL_DIFF", True),
            default_target_url=_env_str("TARGET_URL", "https://books.toscrape.com/"),
            log_level=_env_str("LOG_LEVEL", "INFO"),
            api_host=_env_str("API_HOST", "127.0.0.1"),
            api_port=_env_int("API_PORT", 8000),
            api_base_url=_env_str("API_BASE_URL", "http://127.0.0.1:8000"),
            max_concurrent_runs=_env_int("MAX_CONCURRENT_RUNS", 2),
        )

    # ---------------------------------------------------------------------
    @property
    def llm_available(self) -> bool:
        """True when the pipeline can talk to a model (real or offline stub)."""
        return bool(self.groq_api_key) or self.llm_offline_mode

    def feature_flags(self) -> dict[str, bool]:
        """Flag snapshot recorded into run state and the final report."""
        return {
            "ENABLE_PRD_GAP_ANALYSIS": self.enable_prd_gap_analysis,
            "ENABLE_INTENT_BIAS": self.enable_intent_bias,
            "ENABLE_PARALLEL_EXECUTION": self.enable_parallel_execution,
            "ENABLE_REGRESSION_RADAR": self.enable_regression_radar,
            "ENABLE_VISUAL_DIFF": self.enable_visual_diff,
            "LLM_OFFLINE_MODE": self.llm_offline_mode,
        }

    def safe_dict(self) -> dict[str, Any]:
        """Serialisable settings with the API key elided.

        Used by the report and ``/health``. The key is never included, not
        even truncated.
        """
        out: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if "api_key" in key:
                out[key] = "***SET***" if value else "***UNSET***"
            else:
                out[key] = value
        return out

    def with_overrides(self, **kwargs: Any) -> "Settings":
        """Return a copy with the given fields replaced (used by tests)."""
        return replace(self, **kwargs)


_SETTINGS: Settings | None = None


def get_settings() -> Settings:
    """Process-wide settings singleton."""
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = Settings.from_env()
    return _SETTINGS


def set_settings(settings: Settings) -> None:
    """Replace the singleton. Intended for tests and CLI overrides."""
    global _SETTINGS
    _SETTINGS = settings


def run_dir(run_id: str) -> Path:
    """Directory holding every artifact for a single run."""
    return RUNS_DIR / run_id


def ensure_dirs() -> None:
    """Create the report/baseline directories if they do not exist."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
