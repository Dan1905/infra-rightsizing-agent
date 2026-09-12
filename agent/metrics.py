"""Pull per-container resource metrics from Prometheus (fed by cAdvisor).

Prometheus supplies the utilisation time series; the Docker API supplies the
things cAdvisor does not expose at container granularity -- restart counts,
labels, and lifecycle state. The two are joined on the container name.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import asdict, dataclass, field
from typing import Any

import docker
import requests

from .config import Settings

# rate() needs a range at least 2x the scrape interval to produce a sample.
RATE_WINDOW = "2m"


class MetricsError(RuntimeError):
    """Prometheus was unreachable or answered with an error."""


# --------------------------------------------------------------------------- #
# Prometheus
# --------------------------------------------------------------------------- #


CONTAINER_ID_RE = re.compile(r"([0-9a-f]{64})")


def _instant_query(
    prom_url: str, expr: str, timeout: float = 10.0
) -> list[tuple[dict[str, str], float]]:
    """Run an instant query and return [(labels, value)]."""
    try:
        resp = requests.get(
            f"{prom_url.rstrip('/')}/api/v1/query",
            params={"query": expr},
            timeout=timeout,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise MetricsError(f"Prometheus query failed ({prom_url}): {exc}") from exc

    payload = resp.json()
    if payload.get("status") != "success":
        raise MetricsError(f"Prometheus error for `{expr}`: {payload.get('error')}")

    out: list[tuple[dict[str, str], float]] = []
    for series in payload["data"]["result"]:
        try:
            out.append((series["metric"], float(series["value"][1])))
        except (TypeError, ValueError):
            continue
    return out


def _resolve_name(labels: dict[str, str], id_to_name: dict[str, str]) -> str | None:
    """Map a cAdvisor series onto a container name.

    On hosts where cAdvisor's Docker handler works, series carry a `name`
    label directly. Under the raw cgroup factory (see the note in
    docker-compose.yml) they only carry a cgroup path such as
    `/docker/<64-hex>`, so the id is extracted and looked up against the
    Docker API's inventory. Both are supported.
    """
    name = labels.get("name")
    if name:
        return name
    match = CONTAINER_ID_RE.search(labels.get("id", ""))
    return id_to_name.get(match.group(1)) if match else None


def _queries(lookback: str, step: str) -> dict[str, str]:
    cpu_rate = f"rate(container_cpu_usage_seconds_total[{RATE_WINDOW}])"
    mem = "container_memory_working_set_bytes"
    return {
        "cpu_avg_cores": f"avg_over_time({cpu_rate}[{lookback}:{step}])",
        "cpu_p95_cores": f"quantile_over_time(0.95, {cpu_rate}[{lookback}:{step}])",
        "cpu_max_cores": f"max_over_time({cpu_rate}[{lookback}:{step}])",
        "mem_avg_bytes": f"avg_over_time({mem}[{lookback}])",
        "mem_p95_bytes": f"quantile_over_time(0.95, {mem}[{lookback}])",
        "mem_max_bytes": f"max_over_time({mem}[{lookback}])",
        "mem_limit_bytes": "container_spec_memory_limit_bytes",
        "cpu_quota": "container_spec_cpu_quota",
        "cpu_period": "container_spec_cpu_period",
    }


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


@dataclass
class ContainerMetrics:
    name: str
    image: str
    status: str
    tier: str | None
    managed: bool
    restart_count: int
    uptime_hours: float
    labels: dict[str, str] = field(default_factory=dict)

    cpu_avg_cores: float = 0.0
    cpu_p95_cores: float = 0.0
    cpu_max_cores: float = 0.0
    cpu_limit_cores: float | None = None

    mem_avg_bytes: float = 0.0
    mem_p95_bytes: float = 0.0
    mem_max_bytes: float = 0.0
    mem_limit_bytes: float | None = None

    window: str = "1h"

    # -- derived ------------------------------------------------------------

    @property
    def mem_headroom_ratio(self) -> float | None:
        """Configured limit divided by observed peak. 10.0 means 10x headroom."""
        if not self.mem_limit_bytes or self.mem_max_bytes <= 0:
            return None
        return self.mem_limit_bytes / self.mem_max_bytes

    @property
    def cpu_headroom_ratio(self) -> float | None:
        if not self.cpu_limit_cores or self.cpu_max_cores <= 0:
            return None
        return self.cpu_limit_cores / self.cpu_max_cores

    @property
    def cpu_burstiness(self) -> float | None:
        """peak / mean. ~1 is flat; large values mean the mean is a bad summary."""
        if self.cpu_avg_cores <= 0:
            return None
        return self.cpu_max_cores / self.cpu_avg_cores

    @property
    def mem_burstiness(self) -> float | None:
        if self.mem_avg_bytes <= 0:
            return None
        return self.mem_max_bytes / self.mem_avg_bytes

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.update(
            mem_headroom_ratio=self.mem_headroom_ratio,
            cpu_headroom_ratio=self.cpu_headroom_ratio,
            cpu_burstiness=self.cpu_burstiness,
            mem_burstiness=self.mem_burstiness,
        )
        return data


# --------------------------------------------------------------------------- #
# Collection
# --------------------------------------------------------------------------- #


def _parse_started_at(raw: str | None) -> float:
    if not raw:
        return 0.0
    # Docker returns RFC3339 with nanosecond precision, which fromisoformat
    # rejects on older Pythons -- trim to microseconds.
    cleaned = raw.replace("Z", "+00:00")
    if "." in cleaned:
        head, _, tail = cleaned.partition(".")
        frac, sign, offset = tail.partition("+")
        cleaned = f"{head}.{frac[:6]}{sign}{offset}" if sign else f"{head}.{frac[:6]}"
    try:
        started = dt.datetime.fromisoformat(cleaned)
    except ValueError:
        return 0.0
    if started.tzinfo is None:
        started = started.replace(tzinfo=dt.timezone.utc)
    delta = dt.datetime.now(dt.timezone.utc) - started
    return max(delta.total_seconds() / 3600.0, 0.0)


def collect(settings: Settings, docker_client: docker.DockerClient | None = None) -> list[ContainerMetrics]:
    """Join Docker's container inventory with Prometheus aggregates."""
    client = docker_client or docker.from_env()
    containers = client.containers.list(all=True)
    id_to_name = {c.id: c.name for c in containers}

    series: dict[str, dict[str, float]] = {}
    for key, expr in _queries(settings.lookback, settings.step).items():
        folded: dict[str, float] = {}
        for labels, value in _instant_query(settings.prometheus_url, expr):
            name = _resolve_name(labels, id_to_name)
            if name:
                folded[name] = value
        series[key] = folded

    results: list[ContainerMetrics] = []
    for container in containers:
        name = container.name
        # Stopped containers that aren't ours have no useful series and would
        # only clutter the table; a stopped *managed* container is worth seeing.
        is_managed = (container.labels or {}).get(settings.managed_label, "").lower() == "true"
        if container.status != "running" and not is_managed:
            continue
        labels = container.labels or {}
        attrs = container.attrs or {}

        quota = series["cpu_quota"].get(name, -1.0)
        period = series["cpu_period"].get(name, 0.0)
        cpu_limit = quota / period if quota > 0 and period > 0 else None

        mem_limit = series["mem_limit_bytes"].get(name)
        # cAdvisor reports 0 (and Docker a machine-sized value) for "unlimited".
        if not mem_limit or mem_limit <= 0:
            mem_limit = None

        results.append(
            ContainerMetrics(
                name=name,
                image=(container.image.tags or ["<untagged>"])[0] if container.image else "<none>",
                status=container.status,
                tier=labels.get("cost-opt.tier"),
                managed=is_managed,
                restart_count=int(attrs.get("RestartCount", 0) or 0),
                uptime_hours=_parse_started_at(attrs.get("State", {}).get("StartedAt")),
                labels=labels,
                cpu_avg_cores=series["cpu_avg_cores"].get(name, 0.0),
                cpu_p95_cores=series["cpu_p95_cores"].get(name, 0.0),
                cpu_max_cores=series["cpu_max_cores"].get(name, 0.0),
                cpu_limit_cores=cpu_limit,
                mem_avg_bytes=series["mem_avg_bytes"].get(name, 0.0),
                mem_p95_bytes=series["mem_p95_bytes"].get(name, 0.0),
                mem_max_bytes=series["mem_max_bytes"].get(name, 0.0),
                mem_limit_bytes=mem_limit,
                window=settings.lookback,
            )
        )

    results.sort(key=lambda m: (not m.managed, m.name))
    return results


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def mib(value: float | None) -> str:
    if value is None:
        return "none"
    return f"{value / (1024 * 1024):.0f}MiB"


def format_table(metrics: list[ContainerMetrics]) -> str:
    """A compact fixed-width table -- what gets shown to the operator."""
    header = (
        f"{'CONTAINER':<24} {'TIER':<9} {'CPU avg/p95/max':<22} {'CPU lim':<8} "
        f"{'MEM avg/p95/max':<24} {'MEM lim':<9} {'RSTRT':<6} {'UP(h)':<6}"
    )
    lines = [header, "-" * len(header)]
    for m in metrics:
        cpu = f"{m.cpu_avg_cores:.3f}/{m.cpu_p95_cores:.3f}/{m.cpu_max_cores:.3f}"
        mem = f"{mib(m.mem_avg_bytes)}/{mib(m.mem_p95_bytes)}/{mib(m.mem_max_bytes)}"
        cpu_lim = f"{m.cpu_limit_cores:.2f}" if m.cpu_limit_cores else "none"
        lines.append(
            f"{m.name[:24]:<24} {(m.tier or '-'):<9} {cpu:<22} {cpu_lim:<8} "
            f"{mem:<24} {mib(m.mem_limit_bytes):<9} {m.restart_count:<6} {m.uptime_hours:<6.1f}"
        )
    return "\n".join(lines)


def format_for_llm(metrics: list[ContainerMetrics]) -> str:
    """A denser, self-describing rendering for the model prompt."""
    blocks = []
    for m in metrics:
        headroom_mem = (
            f"{m.mem_headroom_ratio:.1f}x" if m.mem_headroom_ratio is not None else "n/a"
        )
        headroom_cpu = (
            f"{m.cpu_headroom_ratio:.1f}x" if m.cpu_headroom_ratio is not None else "n/a"
        )
        burst_cpu = f"{m.cpu_burstiness:.1f}x" if m.cpu_burstiness is not None else "n/a"
        burst_mem = f"{m.mem_burstiness:.1f}x" if m.mem_burstiness is not None else "n/a"
        blocks.append(
            f"""container: {m.name}
  tier label:        {m.tier or "(none)"}
  image:             {m.image}
  status:            {m.status}
  observation window:{m.window}
  uptime:            {m.uptime_hours:.1f}h
  restarts in window:{m.restart_count}
  CPU cores  avg={m.cpu_avg_cores:.3f}  p95={m.cpu_p95_cores:.3f}  max={m.cpu_max_cores:.3f}
  CPU limit:         {f"{m.cpu_limit_cores:.2f} cores" if m.cpu_limit_cores else "unlimited"}
  CPU limit/peak:    {headroom_cpu}   peak/mean: {burst_cpu}
  MEM  avg={mib(m.mem_avg_bytes)}  p95={mib(m.mem_p95_bytes)}  max={mib(m.mem_max_bytes)}
  MEM limit:         {mib(m.mem_limit_bytes)}
  MEM limit/peak:    {headroom_mem}   peak/mean: {burst_mem}"""
        )
    return "\n\n".join(blocks)
