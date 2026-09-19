"""The data model and contract shared by every execution backend.

A backend knows how to *observe* a set of workloads and how to *change* one.
Everything else -- retrieval, reasoning, proposal validation, human approval,
auditing -- is backend-agnostic and lives outside this package. The safety
checks a backend must run live in guardrails.py.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

MIB = 1024 * 1024


class MetricsError(RuntimeError):
    """The metrics source was unreachable or answered with an error."""


@dataclass
class ExecutionResult:
    ok: bool
    detail: str
    error: str | None = None


# --------------------------------------------------------------------------- #
# Metrics model
# --------------------------------------------------------------------------- #


@dataclass
class WorkloadMetrics:
    """One workload's observed usage and configured resources.

    On Docker a workload is a single container. On Kubernetes it is a
    Deployment, and usage figures are per pod, taken from the hungriest
    replica -- requests and limits are set per pod, so that is the number
    they have to accommodate.
    """

    name: str
    kind: str                      # "container" | "Deployment"
    status: str
    tier: str | None
    managed: bool
    restart_count: int
    # "window" when restart_count covers only the observation window;
    # "lifetime" when it is a since-creation total (Docker's RestartCount).
    restart_scope: str
    uptime_hours: float | None = None
    image: str | None = None
    namespace: str | None = None
    labels: dict[str, str] = field(default_factory=dict)

    cpu_avg_cores: float = 0.0
    cpu_p95_cores: float = 0.0
    cpu_max_cores: float = 0.0
    cpu_limit_cores: float | None = None
    cpu_request_cores: float | None = None

    mem_avg_bytes: float = 0.0
    mem_p95_bytes: float = 0.0
    mem_max_bytes: float = 0.0
    mem_limit_bytes: float | None = None
    mem_request_bytes: float | None = None

    # Kubernetes only.
    replicas: int | None = None
    replicas_available: int | None = None
    hpa_managed: bool | None = None
    last_termination_reason: str | None = None

    window: str = "1h"
    # From the MIN_WINDOW_LABEL label, if the workload declares one.
    min_window: str | None = None

    # -- derived ------------------------------------------------------------

    @staticmethod
    def _ratio(num: float | None, den: float) -> float | None:
        if not num or den <= 0:
            return None
        return num / den

    @property
    def mem_headroom_ratio(self) -> float | None:
        """Configured limit divided by observed peak. 10.0 means 10x headroom."""
        return self._ratio(self.mem_limit_bytes, self.mem_max_bytes)

    @property
    def cpu_headroom_ratio(self) -> float | None:
        return self._ratio(self.cpu_limit_cores, self.cpu_max_cores)

    @property
    def mem_request_ratio(self) -> float | None:
        """Reserved memory divided by observed peak -- the cost-relevant ratio."""
        return self._ratio(self.mem_request_bytes, self.mem_max_bytes)

    @property
    def cpu_request_ratio(self) -> float | None:
        return self._ratio(self.cpu_request_cores, self.cpu_max_cores)

    @property
    def cpu_burstiness(self) -> float | None:
        """peak / mean. ~1 is flat; large values mean the mean is a bad summary."""
        return self._ratio(self.cpu_max_cores, self.cpu_avg_cores)

    @property
    def mem_burstiness(self) -> float | None:
        return self._ratio(self.mem_max_bytes, self.mem_avg_bytes)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for prop in (
            "mem_headroom_ratio", "cpu_headroom_ratio", "mem_request_ratio",
            "cpu_request_ratio", "cpu_burstiness", "mem_burstiness",
        ):
            data[prop] = getattr(self, prop)
        return data


# --------------------------------------------------------------------------- #
# Backend contract
# --------------------------------------------------------------------------- #


class Backend(Protocol):
    name: str
    # Action names this backend can execute, beyond the universal
    # `flag_for_review` and `no_action`. The tool schema offered to the model
    # is built from this, so it is never offered an action it cannot take.
    actions: tuple[str, ...]
    prometheus_url: str

    def collect(self) -> list[WorkloadMetrics]: ...

    def apply(
        self,
        *,
        target: str,
        action: str,
        params: dict[str, Any],
        metrics: WorkloadMetrics | None,
    ) -> ExecutionResult: ...

    def preflight(
        self,
        *,
        target: str,
        action: str,
        params: dict[str, Any],
        metrics: WorkloadMetrics | None,
    ) -> str | None:
        """Run the guardrails against the observed state without changing
        anything. Returns why the change would be blocked, or None.

        Called when the model proposes, so it can correct a proposal before a
        human ever sees it. `apply` runs the same checks again against live
        state, so this is an early warning, never the enforcement point.
        """
        ...

    def describe_scope(self) -> str:
        """One line saying which workloads are eligible, for operator output."""
        ...
