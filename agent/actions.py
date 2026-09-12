"""Execution of approved remediation, plus the guardrails around it.

These checks are deliberately independent of the model. The model can propose
anything; nothing here trusts that proposal. A change is applied only if it
passes every guard *and* a human typed "yes".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import docker
from docker.errors import APIError, NotFound

from .metrics import ContainerMetrics

MIB = 1024 * 1024

# Hard floors, independent of anything the policy corpus says. A retrieved
# document can make the agent more conservative, never less.
MIN_MEMORY_MIB = 128
MIN_CPU_CORES = 0.25
# Refuse to set a memory limit that leaves less than this multiple of the
# observed peak -- the failure mode from the 2026-03-14 postmortem.
MIN_PEAK_MULTIPLE = 1.2


class GuardrailError(RuntimeError):
    """A proposed change failed a safety check and was not applied."""


@dataclass
class ExecutionResult:
    ok: bool
    detail: str
    error: str | None = None


class Executor:
    def __init__(
        self,
        docker_client: docker.DockerClient | None = None,
        *,
        managed_label: str = "cost-opt.managed",
        allow_stop: bool = False,
    ):
        self.client = docker_client or docker.from_env()
        self.managed_label = managed_label
        self.allow_stop = allow_stop

    # -- guards -------------------------------------------------------------

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

    @staticmethod
    def _check_memory(name: str, memory_mib: int, metrics: ContainerMetrics | None) -> None:
        if memory_mib < MIN_MEMORY_MIB:
            raise GuardrailError(
                f"{memory_mib} MiB is below the hard floor of {MIN_MEMORY_MIB} MiB"
            )
        if metrics and metrics.mem_max_bytes > 0:
            floor = (metrics.mem_max_bytes * MIN_PEAK_MULTIPLE) / MIB
            if memory_mib < floor:
                raise GuardrailError(
                    f"{memory_mib} MiB leaves less than {MIN_PEAK_MULTIPLE}x the observed "
                    f"peak of {metrics.mem_max_bytes / MIB:.0f} MiB "
                    f"(minimum {floor:.0f} MiB)"
                )

    @staticmethod
    def _check_cpu(cpu_cores: float) -> None:
        if cpu_cores < MIN_CPU_CORES:
            raise GuardrailError(
                f"{cpu_cores} cores is below the hard floor of {MIN_CPU_CORES} cores"
            )

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

    # -- actions ------------------------------------------------------------

    def apply(
        self,
        *,
        container_name: str,
        action: str,
        params: dict[str, Any],
        metrics: ContainerMetrics | None = None,
    ) -> ExecutionResult:
        try:
            if action in ("no_action", "flag_for_review"):
                return ExecutionResult(
                    ok=True, detail=f"{action}: nothing to execute, recorded in the audit log"
                )

            container = self._require_managed(container_name)

            if action == "set_memory_limit":
                memory_mib = int(params["memory_mib"])
                self._check_memory(container_name, memory_mib, metrics)
                before = metrics.mem_limit_bytes / MIB if metrics and metrics.mem_limit_bytes else None
                container.update(
                    mem_limit=f"{memory_mib}m",
                    memswap_limit=f"{memory_mib}m",
                )
                shown = f"{before:.0f} MiB -> " if before else ""
                return ExecutionResult(
                    ok=True, detail=f"memory limit {shown}{memory_mib} MiB"
                )

            if action == "set_cpu_limit":
                cpu_cores = float(params["cpu_cores"])
                self._check_cpu(cpu_cores)
                self._set_cpu(container, cpu_cores)
                before = f"{metrics.cpu_limit_cores:.2f} -> " if metrics and metrics.cpu_limit_cores else ""
                return ExecutionResult(
                    ok=True, detail=f"CPU limit {before}{cpu_cores:.2f} cores"
                )

            if action == "stop_container":
                if not self.allow_stop:
                    raise GuardrailError(
                        "stop_container is disabled; re-run with --allow-stop to permit it"
                    )
                container.stop(timeout=30)
                return ExecutionResult(ok=True, detail="container stopped")

            raise GuardrailError(f"unknown action `{action}`")

        except GuardrailError as exc:
            return ExecutionResult(ok=False, detail="blocked by guardrail", error=str(exc))
        except (APIError, KeyError, TypeError, ValueError) as exc:
            return ExecutionResult(ok=False, detail="execution failed", error=str(exc))
