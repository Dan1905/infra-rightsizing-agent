"""Kubernetes backend: Deployments in one namespace.

Configuration -- requests, limits, replicas, labels, autoscalers, the last
termination reason -- is read from the Kubernetes API, i.e. from the same
objects the backend patches. Usage over time -- CPU, memory, restarts within
the window -- comes from Prometheus (kube-prometheus-stack: cAdvisor through
the kubelet, plus kube-state-metrics).

Usage is recorded per pod and summarised per Deployment from its busiest
replica, because requests and limits are set per pod.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Any

from kubernetes import client as k8s
from kubernetes import config as k8s_config
from kubernetes.client.rest import ApiException
from kubernetes.utils import parse_quantity

from ..config import Settings
from .base import ExecutionResult, WorkloadMetrics
from .guardrails import (
    GuardrailError,
    MIN_REPLICAS,
    MIN_WINDOW_LABEL,
    check_cpu,
    check_cpu_request,
    check_memory,
    check_memory_request,
    check_window,
)
from .prometheus import instant_query

RATE_WINDOW = "2m"


def _mem_bytes(q: str | None) -> float | None:
    return float(parse_quantity(q)) if q else None


def _cores(q: str | None) -> float | None:
    return float(parse_quantity(q)) if q else None


def _mem_quantity(mib: int) -> str:
    return f"{mib}Mi"


def _cpu_quantity(cores: float) -> str:
    return f"{int(round(cores * 1000))}m"


class KubernetesBackend:
    name = "kubernetes"
    actions = ("set_requests", "set_limits", "scale_replicas")

    def __init__(self, settings: Settings):
        self.settings = settings
        self.prometheus_url = settings.prometheus_url
        self.namespace = settings.k8s_namespace
        self.managed_label = settings.managed_label
        k8s_config.load_kube_config(context=settings.kube_context or None)
        self.apps = k8s.AppsV1Api()
        self.core = k8s.CoreV1Api()
        self.autoscaling = k8s.AutoscalingV2Api()

    def describe_scope(self) -> str:
        return f"namespace {self.namespace}, label {self.managed_label}=true"

    # -- observe ------------------------------------------------------------

    def _pod_series(self, expr: str) -> dict[tuple[str, str], float]:
        """{(pod, container): value} for a query grouped by pod and container."""
        out: dict[tuple[str, str], float] = {}
        for labels, value in instant_query(self.prometheus_url, expr):
            pod, container = labels.get("pod"), labels.get("container")
            if pod and container:
                out[(pod, container)] = value
        return out

    def _usage(self) -> dict[str, dict[tuple[str, str], float]]:
        s, ns = self.settings, self.namespace
        sel = f'namespace="{ns}", container!="", container!="POD"'
        cpu = f"rate(container_cpu_usage_seconds_total{{{sel}}}[{RATE_WINDOW}])"
        mem = f"container_memory_working_set_bytes{{{sel}}}"
        by = "by (pod, container)"
        queries = {
            "cpu_avg": f"max {by} (avg_over_time({cpu}[{s.lookback}:{s.step}]))",
            "cpu_p95": f"max {by} (quantile_over_time(0.95, {cpu}[{s.lookback}:{s.step}]))",
            "cpu_max": f"max {by} (max_over_time({cpu}[{s.lookback}:{s.step}]))",
            "mem_avg": f"max {by} (avg_over_time({mem}[{s.lookback}]))",
            "mem_p95": f"max {by} (quantile_over_time(0.95, {mem}[{s.lookback}]))",
            "mem_max": f"max {by} (max_over_time({mem}[{s.lookback}]))",
            # Unlike Docker's RestartCount, this is genuinely restarts within
            # the window.
            "restarts": (
                f"sum {by} (increase(kube_pod_container_status_restarts_total"
                f'{{namespace="{ns}"}}[{s.lookback}]))'
            ),
        }
        return {key: self._pod_series(expr) for key, expr in queries.items()}

    def collect(self) -> list[WorkloadMetrics]:
        usage = self._usage()
        hpa_targets = {
            h.spec.scale_target_ref.name
            for h in self.autoscaling.list_namespaced_horizontal_pod_autoscaler(
                self.namespace
            ).items
            if h.spec.scale_target_ref.kind == "Deployment"
        }
        pods = self.core.list_namespaced_pod(self.namespace).items

        results: list[WorkloadMetrics] = []
        for dep in self.apps.list_namespaced_deployment(self.namespace).items:
            name = dep.metadata.name
            labels = dep.metadata.labels or {}
            container = dep.spec.template.spec.containers[0]
            resources = container.resources
            requests = (resources.requests or {}) if resources else {}
            limits = (resources.limits or {}) if resources else {}

            # Deployment pods are named <deployment>-<replicaset hash>-<suffix>.
            # Matching on the name (not just current pods) keeps pods replaced
            # during the window -- e.g. after a crash -- in the figures.
            pod_re = re.compile(rf"^{re.escape(name)}-[a-z0-9]+-[a-z0-9]{{5}}$")

            def values(key: str) -> list[float]:
                return [
                    v for (pod, cname), v in usage[key].items()
                    if cname == container.name and pod_re.match(pod)
                ]

            def busiest(key: str) -> float:
                vals = values(key)
                return max(vals) if vals else 0.0

            def mean(key: str) -> float:
                vals = values(key)
                return sum(vals) / len(vals) if vals else 0.0

            own_pods = [p for p in pods if pod_re.match(p.metadata.name)]
            last_reason = None
            started: list[dt.datetime] = []
            for pod in own_pods:
                if pod.status.start_time:
                    started.append(pod.status.start_time)
                for cs in pod.status.container_statuses or []:
                    term = cs.last_state.terminated if cs.last_state else None
                    if cs.name == container.name and term and term.reason:
                        # OOMKilled is the reason that matters most for sizing.
                        if last_reason != "OOMKilled":
                            last_reason = term.reason
            uptime = None
            if started:
                oldest = min(started)
                uptime = (dt.datetime.now(dt.timezone.utc) - oldest).total_seconds() / 3600

            desired = dep.spec.replicas or 0
            available = dep.status.available_replicas or 0

            results.append(
                WorkloadMetrics(
                    name=name,
                    kind="Deployment",
                    namespace=self.namespace,
                    image=container.image,
                    status="available" if available >= desired else "degraded",
                    tier=labels.get("cost-opt.tier"),
                    managed=labels.get(self.managed_label, "").lower() == "true",
                    restart_count=int(round(sum(values("restarts")))),
                    restart_scope="window",
                    uptime_hours=uptime,
                    labels=labels,
                    cpu_avg_cores=mean("cpu_avg"),
                    cpu_p95_cores=busiest("cpu_p95"),
                    cpu_max_cores=busiest("cpu_max"),
                    cpu_limit_cores=_cores(limits.get("cpu")),
                    cpu_request_cores=_cores(requests.get("cpu")),
                    mem_avg_bytes=mean("mem_avg"),
                    mem_p95_bytes=busiest("mem_p95"),
                    mem_max_bytes=busiest("mem_max"),
                    mem_limit_bytes=_mem_bytes(limits.get("memory")),
                    mem_request_bytes=_mem_bytes(requests.get("memory")),
                    replicas=desired,
                    replicas_available=available,
                    hpa_managed=name in hpa_targets,
                    last_termination_reason=last_reason,
                    window=self.settings.lookback,
                    min_window=labels.get(MIN_WINDOW_LABEL),
                )
            )

        results.sort(key=lambda m: (not m.managed, m.name))
        return results

    # -- change -------------------------------------------------------------

    def _require_managed(self, name: str):
        """Re-read the live Deployment and check it is ours to change."""
        try:
            dep = self.apps.read_namespaced_deployment(name, self.namespace)
        except ApiException as exc:
            if exc.status == 404:
                raise GuardrailError(
                    f"Deployment `{name}` does not exist in namespace {self.namespace}"
                ) from exc
            raise
        if (dep.metadata.labels or {}).get(self.managed_label, "").lower() != "true":
            raise GuardrailError(
                f"Deployment `{name}` is not labelled {self.managed_label}=true; "
                "the agent may not modify it"
            )
        if len(dep.spec.template.spec.containers) != 1:
            raise GuardrailError(
                f"Deployment `{name}` runs {len(dep.spec.template.spec.containers)} "
                "containers per pod; only single-container pods are supported"
            )
        return dep

    def _patch(self, name: str, body: dict[str, Any]) -> None:
        """Server-side dry run first, so the API server validates the change
        (schema, quotas, admission policies) before anything is persisted."""
        try:
            self.apps.patch_namespaced_deployment(
                name, self.namespace, body, dry_run="All"
            )
        except ApiException as exc:
            raise GuardrailError(
                f"the API server rejected the change in a dry run: {exc.reason}: "
                f"{(exc.body or '')[:300]}"
            ) from exc
        self.apps.patch_namespaced_deployment(name, self.namespace, body)

    @staticmethod
    def _new_resources(
        kind: str,
        params: dict[str, Any],
        metrics: WorkloadMetrics | None,
        requests: dict[str, str],
        limits: dict[str, str],
    ) -> tuple[dict[str, str], list[str]]:
        """Apply `params` to the requests or limits and run every resource
        guardrail. Returns the new values for `kind` and a change summary."""
        requests, limits = dict(requests), dict(limits)
        target = requests if kind == "requests" else limits

        changes = []
        mem, cpu = params.get("memory_mib"), params.get("cpu_cores")
        if mem is not None:
            if kind == "requests":
                check_memory_request(int(mem), metrics)
            else:
                check_memory(int(mem), metrics)
            changes.append(f"memory {target.get('memory', 'unset')} -> {_mem_quantity(int(mem))}")
            target["memory"] = _mem_quantity(int(mem))
        if cpu is not None:
            if kind == "requests":
                check_cpu_request(float(cpu))
            else:
                check_cpu(float(cpu))
            changes.append(f"cpu {target.get('cpu', 'unset')} -> {_cpu_quantity(float(cpu))}")
            target["cpu"] = _cpu_quantity(float(cpu))

        # Kubernetes rejects requests above limits; say so plainly first.
        for res in ("memory", "cpu"):
            if res in requests and res in limits:
                if parse_quantity(requests[res]) > parse_quantity(limits[res]):
                    raise GuardrailError(
                        f"{res} request {requests[res]} would exceed the {res} "
                        f"limit {limits[res]}"
                    )
        return target, changes

    def _check_scale(self, target: str, replicas: int, metrics: WorkloadMetrics | None) -> None:
        if replicas < MIN_REPLICAS:
            raise GuardrailError(
                f"{replicas} replicas is below the floor of {MIN_REPLICAS}; "
                "scaling to zero is stopping the service"
            )
        if metrics and metrics.hpa_managed:
            raise GuardrailError(
                f"a HorizontalPodAutoscaler manages `{target}`; changing "
                "replicas by hand would fight it -- change the HPA instead"
            )

    @staticmethod
    def _observed_resources(metrics: WorkloadMetrics) -> tuple[dict[str, str], dict[str, str]]:
        """Requests and limits as quantity strings, from a metrics snapshot."""
        def q(mem: float | None, cpu: float | None) -> dict[str, str]:
            out = {}
            if mem is not None:
                out["memory"] = str(int(mem))
            if cpu is not None:
                out["cpu"] = _cpu_quantity(cpu)
            return out
        return (
            q(metrics.mem_request_bytes, metrics.cpu_request_cores),
            q(metrics.mem_limit_bytes, metrics.cpu_limit_cores),
        )

    def preflight(
        self,
        *,
        target: str,
        action: str,
        params: dict[str, Any],
        metrics: WorkloadMetrics | None,
    ) -> str | None:
        try:
            check_window(metrics)
            if action in ("set_requests", "set_limits") and metrics is not None:
                requests, limits = self._observed_resources(metrics)
                kind = "requests" if action == "set_requests" else "limits"
                self._new_resources(kind, params, metrics, requests, limits)
            elif action == "scale_replicas":
                self._check_scale(target, int(params["replicas"]), metrics)
        except GuardrailError as exc:
            return str(exc)
        return None

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

            dep = self._require_managed(target)
            check_window(metrics)

            if action in ("set_requests", "set_limits"):
                kind = "requests" if action == "set_requests" else "limits"
                container = dep.spec.template.spec.containers[0]
                current = container.resources
                target_values, changes = self._new_resources(
                    kind,
                    params,
                    metrics,
                    (current.requests or {}) if current else {},
                    (current.limits or {}) if current else {},
                )
                self._patch(target, {"spec": {"template": {"spec": {"containers": [
                    {"name": container.name, "resources": {kind: target_values}}
                ]}}}})
                return ExecutionResult(
                    ok=True, detail=f"{kind}: {', '.join(changes)} (rolling out new pods)"
                )

            if action == "scale_replicas":
                replicas = int(params["replicas"])
                self._check_scale(target, replicas, metrics)
                before = dep.spec.replicas
                self._patch(target, {"spec": {"replicas": replicas}})
                return ExecutionResult(ok=True, detail=f"replicas {before} -> {replicas}")

            raise GuardrailError(f"action `{action}` is not supported by the kubernetes backend")

        except GuardrailError as exc:
            return ExecutionResult(ok=False, detail="blocked by guardrail", error=str(exc))
        except (ApiException, KeyError, TypeError, ValueError) as exc:
            return ExecutionResult(ok=False, detail="execution failed", error=str(exc))
