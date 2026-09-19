"""Rendering workload metrics: a table for the operator, labelled blocks for the model."""

from __future__ import annotations

from .backends.base import MIB, WorkloadMetrics


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
