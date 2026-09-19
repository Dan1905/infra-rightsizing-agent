"""Docker backend: cAdvisor metrics via Prometheus, changes via the Docker SDK.

Prometheus supplies the utilisation time series; the Docker API supplies the
things cAdvisor does not expose at container granularity -- restart counts,
labels, and lifecycle state. The two are joined on the container name.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Any

import docker
from docker.errors import APIError, NotFound

from ..config import Settings
from .base import (
    MIB,
    ExecutionResult,
    GuardrailError,
    WorkloadMetrics,
    check_cpu,
    check_memory,
)
from .prometheus import instant_query

# rate() needs a range at least 2x the scrape interval to produce a sample.
RATE_WINDOW = "2m"
CONTAINER_ID_RE = re.compile(r"([0-9a-f]{64})")


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


class DockerBackend:
    name = "docker"
    actions = ("set_memory_limit", "set_cpu_limit", "stop_container")

    def __init__(
        self,
        settings: Settings,
        docker_client: docker.DockerClient | None = None,
        *,
        allow_stop: bool = False,
    ):
        self.settings = settings
        self.prometheus_url = settings.prometheus_url
        self.managed_label = settings.managed_label
        self.allow_stop = allow_stop
        self.client = docker_client or docker.from_env()

    def describe_scope(self) -> str:
        return f"label {self.managed_label}=true"

    # -- observe ------------------------------------------------------------

    def collect(self) -> list[WorkloadMetrics]:
        """Join Docker's container inventory with Prometheus aggregates."""
        s = self.settings
        containers = self.client.containers.list(all=True)
        id_to_name = {c.id: c.name for c in containers}

        series: dict[str, dict[str, float]] = {}
        for key, expr in _queries(s.lookback, s.step).items():
            folded: dict[str, float] = {}
            for labels, value in instant_query(self.prometheus_url, expr):
                name = _resolve_name(labels, id_to_name)
                if name:
                    folded[name] = value
            series[key] = folded

        results: list[WorkloadMetrics] = []
        for container in containers:
            name = container.name
            labels = container.labels or {}
            # Stopped containers that aren't ours have no useful series and
            # would only clutter the table; a stopped *managed* one is worth seeing.
            is_managed = labels.get(self.managed_label, "").lower() == "true"
            if container.status != "running" and not is_managed:
                continue
            attrs = container.attrs or {}

            quota = series["cpu_quota"].get(name, -1.0)
            period = series["cpu_period"].get(name, 0.0)
            cpu_limit = quota / period if quota > 0 and period > 0 else None

            mem_limit = series["mem_limit_bytes"].get(name)
            # cAdvisor reports 0 for "unlimited".
            if not mem_limit or mem_limit <= 0:
                mem_limit = None

            results.append(
                WorkloadMetrics(
                    name=name,
                    kind="container",
                    image=(container.image.tags or ["<untagged>"])[0] if container.image else None,
                    status=container.status,
                    tier=labels.get("cost-opt.tier"),
                    managed=is_managed,
                    # Docker only knows the since-creation total, and says so.
                    restart_count=int(attrs.get("RestartCount", 0) or 0),
                    restart_scope="lifetime",
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
                    window=s.lookback,
                )
            )

        results.sort(key=lambda m: (not m.managed, m.name))
        return results

    # -- change -------------------------------------------------------------

    def _require_managed(self, name: str):
        try:
            container = self.client.containers.get(name)
        except NotFound as exc:
            raise GuardrailError(f"container `{name}` does not exist") from exc
        if (container.labels or {}).get(self.managed_label, "").lower() != "true":
            raise GuardrailError(
                f"container `{name}` is not labelled {self.managed_label}=true; "
                "the agent may not modify it"
            )
        return container

    def _set_cpu(self, container, cpu_cores: float) -> None:
        """Set a CPU limit, whichever way the container was originally created.

        A container created with `--cpus` / compose's `cpus:` carries NanoCpus,
        and the daemon rejects a CpuQuota/CpuPeriod update while that is set
        ("Conflicting options"). docker-py's `update()` does not expose
        NanoCpus, so that case posts the update directly.
        """
        if (container.attrs.get("HostConfig") or {}).get("NanoCpus"):
            api = self.client.api
            api._raise_for_status(
                api._post_json(
                    api._url("/containers/{0}/update", container.id),
                    data={"NanoCpus": int(cpu_cores * 1_000_000_000)},
                )
            )
        else:
            period = 100_000
            container.update(cpu_period=period, cpu_quota=int(cpu_cores * period))

    def apply(
        self,
        *,
        target: str,
        action: str,
        params: dict[str, Any],
        metrics: WorkloadMetrics | None = None,
    ) -> ExecutionResult:
        try:
            if action in ("no_action", "flag_for_review"):
                return ExecutionResult(
                    ok=True, detail=f"{action}: nothing to execute, recorded in the audit log"
                )

            container = self._require_managed(target)

            if action == "set_memory_limit":
                memory_mib = int(params["memory_mib"])
                check_memory(memory_mib, metrics)
                before = metrics.mem_limit_bytes / MIB if metrics and metrics.mem_limit_bytes else None
                container.update(mem_limit=f"{memory_mib}m", memswap_limit=f"{memory_mib}m")
                shown = f"{before:.0f} MiB -> " if before else ""
                return ExecutionResult(ok=True, detail=f"memory limit {shown}{memory_mib} MiB")

            if action == "set_cpu_limit":
                cpu_cores = float(params["cpu_cores"])
                check_cpu(cpu_cores)
                self._set_cpu(container, cpu_cores)
                before = f"{metrics.cpu_limit_cores:.2f} -> " if metrics and metrics.cpu_limit_cores else ""
                return ExecutionResult(ok=True, detail=f"CPU limit {before}{cpu_cores:.2f} cores")

            if action == "stop_container":
                if not self.allow_stop:
                    raise GuardrailError(
                        "stop_container is disabled; re-run with --allow-stop to permit it"
                    )
                container.stop(timeout=30)
                return ExecutionResult(ok=True, detail="container stopped")

            raise GuardrailError(f"action `{action}` is not supported by the docker backend")

        except GuardrailError as exc:
            return ExecutionResult(ok=False, detail="blocked by guardrail", error=str(exc))
        except (APIError, KeyError, TypeError, ValueError) as exc:
            return ExecutionResult(ok=False, detail="execution failed", error=str(exc))
