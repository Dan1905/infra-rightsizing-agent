"""What every execution backend shares: the metrics model, the guardrail
primitives, and the contract a backend must meet.

A backend knows how to *observe* a set of workloads and how to *change* one.
Everything else -- retrieval, reasoning, proposal validation, human approval,
auditing -- is backend-agnostic and lives outside this package.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

MIB = 1024 * 1024

# Hard floors, independent of anything the policy corpus says. A retrieved
# document can make the agent more conservative, never less.
MIN_MEMORY_MIB = 128
MIN_CPU_CORES = 0.25
# Refuse to set a memory limit (or request) that leaves less than this multiple
# of the observed peak -- the failure mode from the 2026-03-14 postmortem.
MIN_PEAK_MULTIPLE = 1.2
# Requests are reservations, not ceilings: a pod using more than its request is
# not killed for it. They may therefore go lower than limits -- but not to zero.
MIN_REQUEST_MEMORY_MIB = 32
MIN_REQUEST_CPU_CORES = 0.05
MIN_REPLICAS = 1


# A workload can declare how much history a sizing decision about it needs --
# e.g. "24h" for a job that only does real work once a night. Rightsizing it on
# less is the failure mode from the 2026-03-14 postmortem, so it is enforced
# here rather than left to the model reading the right document.
MIN_WINDOW_LABEL = "cost-opt.min-window"

_DURATION_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


def parse_duration(value: str) -> int:
    """Seconds in a Prometheus-style duration such as 30m, 1h, 1h30m, 7d."""
    import re

    parts = re.findall(r"(\d+)([smhdw])", value.strip())
    if not parts or "".join(n + u for n, u in parts) != value.strip():
        raise ValueError(f"not a duration: {value!r}")
    return sum(int(n) * _DURATION_UNITS[u] for n, u in parts)


class MetricsError(RuntimeError):
    """The metrics source was unreachable or answered with an error."""


class GuardrailError(RuntimeError):
    """A proposed change failed a safety check and was not applied."""


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
# Shared guardrails
# --------------------------------------------------------------------------- #


def check_memory(memory_mib: int, metrics: WorkloadMetrics | None, *, what: str = "limit") -> None:
    if memory_mib < MIN_MEMORY_MIB:
        raise GuardrailError(
            f"{memory_mib} MiB is below the hard floor of {MIN_MEMORY_MIB} MiB"
        )
    if metrics and metrics.mem_max_bytes > 0:
        floor = (metrics.mem_max_bytes * MIN_PEAK_MULTIPLE) / MIB
        if memory_mib < floor:
            raise GuardrailError(
                f"a {memory_mib} MiB memory {what} leaves less than {MIN_PEAK_MULTIPLE}x "
                f"the observed peak of {metrics.mem_max_bytes / MIB:.0f} MiB "
                f"(minimum {floor:.0f} MiB)"
            )


def check_window(metrics: WorkloadMetrics | None) -> None:
    """Refuse any change to a workload observed for less than it declares it needs."""
    if not metrics or not metrics.min_window:
        return
    try:
        needed = parse_duration(metrics.min_window)
    except ValueError:
        raise GuardrailError(
            f"`{metrics.name}` declares an unreadable {MIN_WINDOW_LABEL} "
            f"({metrics.min_window!r}); refusing to change it until that is fixed"
        )
    if parse_duration(metrics.window) < needed:
        raise GuardrailError(
            f"`{metrics.name}` declares that sizing decisions need at least "
            f"{metrics.min_window} of observation ({MIN_WINDOW_LABEL}); this run "
            f"observed {metrics.window}. Leave it unchanged, or re-run with "
            f"--lookback {metrics.min_window} once that much history exists"
        )


def check_cpu(cpu_cores: float) -> None:
    if cpu_cores < MIN_CPU_CORES:
        raise GuardrailError(
            f"{cpu_cores} cores is below the hard floor of {MIN_CPU_CORES} cores"
        )


def check_memory_request(memory_mib: int, metrics: WorkloadMetrics | None) -> None:
    if memory_mib < MIN_REQUEST_MEMORY_MIB:
        raise GuardrailError(
            f"a {memory_mib} MiB memory request is below the floor of "
            f"{MIN_REQUEST_MEMORY_MIB} MiB"
        )
    # A request under the observed peak means the scheduler reserves less than
    # the pod demonstrably uses -- the node can then be overcommitted into OOM.
    if metrics and metrics.mem_max_bytes > 0:
        floor = (metrics.mem_max_bytes * MIN_PEAK_MULTIPLE) / MIB
        if memory_mib < floor:
            raise GuardrailError(
                f"a {memory_mib} MiB memory request leaves less than "
                f"{MIN_PEAK_MULTIPLE}x the observed peak of "
                f"{metrics.mem_max_bytes / MIB:.0f} MiB (minimum {floor:.0f} MiB)"
            )


def check_cpu_request(cpu_cores: float) -> None:
    if cpu_cores < MIN_REQUEST_CPU_CORES:
        raise GuardrailError(
            f"a {cpu_cores} core CPU request is below the floor of "
            f"{MIN_REQUEST_CPU_CORES} cores"
        )


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


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def mib(value: float | None) -> str:
    if value is None:
        return "none"
    return f"{value / MIB:.0f}MiB"


def _cores(value: float | None) -> str:
    return f"{value:.2f}" if value else "none"


def format_table(metrics: list[WorkloadMetrics]) -> str:
    """A compact fixed-width table -- what gets shown to the operator.

    Request and replica columns appear only when some workload has them, so
    the Docker view is unchanged.
    """
    k8s = any(m.cpu_request_cores or m.mem_request_bytes or m.replicas for m in metrics)
    cols = [
        ("WORKLOAD", 24), ("TIER", 9), ("CPU avg/p95/max", 22),
        ("CPU req/lim" if k8s else "CPU lim", 12 if k8s else 8),
        ("MEM avg/p95/max", 24),
        ("MEM req/lim" if k8s else "MEM lim", 17 if k8s else 9),
    ]
    if k8s:
        cols.append(("REPL", 6))
    cols += [("RSTRT", 6), ("UP(h)", 6)]

    header = " ".join(f"{title:<{w}}" for title, w in cols)
    lines = [header, "-" * len(header)]
    for m in metrics:
        cpu = f"{m.cpu_avg_cores:.3f}/{m.cpu_p95_cores:.3f}/{m.cpu_max_cores:.3f}"
        mem = f"{mib(m.mem_avg_bytes)}/{mib(m.mem_p95_bytes)}/{mib(m.mem_max_bytes)}"
        if k8s:
            cpu_cfg = f"{_cores(m.cpu_request_cores)}/{_cores(m.cpu_limit_cores)}"
            mem_cfg = f"{mib(m.mem_request_bytes)}/{mib(m.mem_limit_bytes)}"
        else:
            cpu_cfg, mem_cfg = _cores(m.cpu_limit_cores), mib(m.mem_limit_bytes)
        restarts = f"{m.restart_count}" + ("*" if m.restart_scope == "lifetime" else "")
        up = f"{m.uptime_hours:.1f}" if m.uptime_hours is not None else "-"
        values = [m.name[:24], m.tier or "-", cpu, cpu_cfg, mem, mem_cfg]
        if k8s:
            repl = (
                f"{m.replicas_available}/{m.replicas}" if m.replicas is not None else "-"
            )
            values.append(repl)
        values += [restarts, up]
        lines.append(" ".join(f"{v:<{w}}" for v, (_, w) in zip(values, cols)))
    if any(m.restart_scope == "lifetime" for m in metrics):
        lines.append("* restarts since the container was created, not within the window")
    return "\n".join(lines)


def _ratio_text(value: float | None) -> str:
    return f"{value:.1f}x" if value is not None else "n/a"


def format_for_llm(metrics: list[WorkloadMetrics]) -> str:
    """A denser, self-describing rendering for the model prompt."""
    blocks = []
    for m in metrics:
        restart_label = (
            f"restarts in window:  {m.restart_count}"
            if m.restart_scope == "window"
            else f"restarts (lifetime): {m.restart_count}  "
                 "(since container creation -- NOT limited to the window)"
        )
        lines = [
            f"workload: {m.name}",
            f"  kind:               {m.kind}"
            + (f" in namespace {m.namespace}" if m.namespace else ""),
            f"  tier label:         {m.tier or '(none)'}",
            f"  status:             {m.status}",
            f"  observation window: {m.window}",
        ]
        if m.uptime_hours is not None:
            lines.append(f"  uptime:             {m.uptime_hours:.1f}h")
        if m.replicas is not None:
            lines.append(
                f"  replicas:           {m.replicas} desired, "
                f"{m.replicas_available} available"
            )
        if m.hpa_managed is not None:
            lines.append(
                "  autoscaler:         "
                + ("HPA manages the replica count" if m.hpa_managed else "none")
            )
        if m.min_window:
            lines.append(
                f"  declared min window: {m.min_window}  (sizing decisions need at "
                f"least this much history; this run observed {m.window})"
            )
        lines.append(f"  {restart_label}")
        if m.last_termination_reason:
            lines.append(f"  last termination:   {m.last_termination_reason}")
        if m.kind == "Deployment":
            lines.append("  usage figures are per pod, from the busiest replica")

        lines += [
            f"  CPU cores  avg={m.cpu_avg_cores:.3f}  p95={m.cpu_p95_cores:.3f}  "
            f"max={m.cpu_max_cores:.3f}",
        ]
        if m.cpu_request_cores is not None:
            lines.append(
                f"  CPU request:        {m.cpu_request_cores:.2f} cores   "
                f"request/peak: {_ratio_text(m.cpu_request_ratio)}"
            )
        lines += [
            f"  CPU limit:          "
            + (f"{m.cpu_limit_cores:.2f} cores" if m.cpu_limit_cores else "unlimited")
            + f"   limit/peak: {_ratio_text(m.cpu_headroom_ratio)}"
            + f"   peak/mean: {_ratio_text(m.cpu_burstiness)}",
            f"  MEM  avg={mib(m.mem_avg_bytes)}  p95={mib(m.mem_p95_bytes)}  "
            f"max={mib(m.mem_max_bytes)}",
        ]
        if m.mem_request_bytes is not None:
            lines.append(
                f"  MEM request:        {mib(m.mem_request_bytes)}   "
                f"request/peak: {_ratio_text(m.mem_request_ratio)}"
            )
        lines.append(
            f"  MEM limit:          {mib(m.mem_limit_bytes)}"
            f"   limit/peak: {_ratio_text(m.mem_headroom_ratio)}"
            f"   peak/mean: {_ratio_text(m.mem_burstiness)}"
        )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
