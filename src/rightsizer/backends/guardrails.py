"""Every hard safety check, in one place.

These run twice -- when the model proposes (preflight, against the observed
state) and again when an approved change is executed (against live state) --
and they are independent of anything the model or the policy corpus says.
"""

from __future__ import annotations

import re

from .base import MIB, WorkloadMetrics


class GuardrailError(RuntimeError):
    """A proposed change failed a safety check and was not applied."""


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
    parts = re.findall(r"(\d+)([smhdw])", value.strip())
    if not parts or "".join(n + u for n, u in parts) != value.strip():
        raise ValueError(f"not a duration: {value!r}")
    return sum(int(n) * _DURATION_UNITS[u] for n, u in parts)


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
