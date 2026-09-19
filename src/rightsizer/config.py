"""Runtime configuration, read from the environment with sane local defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# src/rightsizer/config.py -> repository root. Paths default to the checkout
# (policies/, data/, .env); every one can be overridden from the environment.
REPO_ROOT = Path(__file__).resolve().parents[2]

# Load .env before any field default reads the environment. Real environment
# variables win over the file, so `GROQ_API_KEY=... rightsize analyze` still
# overrides whatever .env holds.
try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env", override=False)
except ImportError:  # pragma: no cover - .env is optional
    pass


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    return int(raw) if raw else default


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    return float(raw) if raw else default


@dataclass(frozen=True)
class Settings:
    # --- Backend ------------------------------------------------------------
    # docker (default: the compose sandbox) | kubernetes (the minikube sandbox).
    backend: str = field(default_factory=lambda: _env_str("BACKEND", "docker").lower())

    # --- Metrics source -----------------------------------------------------
    # The two sandboxes run separate Prometheus instances; the Kubernetes one is
    # reached through `kubectl port-forward` on 9091 (see `make port-forward`).
    prometheus_url: str = field(
        default_factory=lambda: _env_str(
            "PROMETHEUS_URL",
            "http://localhost:9091"
            if os.environ.get("BACKEND", "docker").lower() == "kubernetes"
            else "http://localhost:9090",
        )
    )
    # Observation window used for every aggregate. Deliberately configurable:
    # the policy corpus has opinions about windows that are too short.
    lookback: str = field(default_factory=lambda: _env_str("LOOKBACK", "1h"))
    # Resolution of the inner rate() subquery.
    step: str = field(default_factory=lambda: _env_str("METRICS_STEP", "1m"))

    # --- Scope --------------------------------------------------------------
    # Only containers carrying this label are eligible for proposals. This is
    # the hard boundary that keeps the agent away from its own stack.
    managed_label: str = field(
        default_factory=lambda: _env_str("MANAGED_LABEL", "cost-opt.managed")
    )
    # Kubernetes only: the single namespace the agent may observe and change,
    # and the kubeconfig context to use (empty = current context).
    k8s_namespace: str = field(
        default_factory=lambda: _env_str("K8S_NAMESPACE", "cost-opt-sandbox")
    )
    kube_context: str = field(default_factory=lambda: _env_str("KUBE_CONTEXT", "cost-opt"))

    # --- RAG ----------------------------------------------------------------
    policies_dir: Path = field(
        default_factory=lambda: Path(_env_str("POLICIES_DIR", str(REPO_ROOT / "policies")))
    )
    chroma_dir: Path = field(
        default_factory=lambda: Path(_env_str("CHROMA_DIR", str(REPO_ROOT / "data" / "chroma")))
    )
    collection_name: str = field(
        default_factory=lambda: _env_str("CHROMA_COLLECTION", "policies")
    )
    embedding_model: str = field(
        default_factory=lambda: _env_str("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
    top_k: int = field(default_factory=lambda: _env_int("RAG_TOP_K", 3))

    # --- LLM ----------------------------------------------------------------
    # groq (default, free tier, gpt-oss-120b) | anthropic | openai.
    # `openai` is the generic OpenAI-compatible slot -- point LLM_BASE_URL at
    # xAI, Together, OpenRouter, a local vLLM, or anything else that speaks
    # chat-completions, and set LLM_MODEL.
    provider: str = field(default_factory=lambda: _env_str("LLM_PROVIDER", "groq").lower())
    # Empty means "use the provider's default" (see providers.PRESETS).
    model: str = field(default_factory=lambda: _env_str("LLM_MODEL", ""))
    base_url: str = field(default_factory=lambda: _env_str("LLM_BASE_URL", ""))
    api_key: str = field(default_factory=lambda: _env_str("LLM_API_KEY", ""))

    # 0 = the provider's default (see providers.PRESETS).
    max_tokens: int = field(default_factory=lambda: _env_int("MAX_TOKENS", 0))
    temperature: float = field(default_factory=lambda: _env_float("TEMPERATURE", 0.2))
    max_retries: int = field(default_factory=lambda: _env_int("MAX_RETRIES", 4))
    max_turns: int = field(default_factory=lambda: _env_int("MAX_TURNS", 12))
    # Anthropic-only; ignored by the OpenAI-compatible providers.
    effort: str = field(default_factory=lambda: _env_str("EFFORT", "high"))
    # Reasoning models on OpenAI-compatible endpoints (gpt-oss): low | medium |
    # high. Empty = the provider's default (see providers.py).
    reasoning_effort: str = field(default_factory=lambda: _env_str("REASONING_EFFORT", ""))

    # --- Audit --------------------------------------------------------------
    audit_db: Path = field(
        default_factory=lambda: Path(_env_str("AUDIT_DB", str(REPO_ROOT / "data" / "audit.db")))
    )


settings = Settings()
