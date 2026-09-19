"""KubernetesBackend.apply/preflight against a fake API -- no cluster needed."""

import pytest
from kubernetes import client as k8s
from kubernetes.client.rest import ApiException

from agent.backends.base import MIB
from agent.backends.kubernetes import KubernetesBackend

from .conftest import make_metrics

MANAGED = {"cost-opt.managed": "true"}


def deployment(name, labels, requests, limits, replicas=2, containers=1):
    specs = [
        k8s.V1Container(
            name=name if i == 0 else f"sidecar-{i}",
            image="python:3.12-alpine",
            resources=k8s.V1ResourceRequirements(requests=dict(requests), limits=dict(limits)),
        )
        for i in range(containers)
    ]
    return k8s.V1Deployment(
        metadata=k8s.V1ObjectMeta(name=name, labels=labels),
        spec=k8s.V1DeploymentSpec(
            replicas=replicas,
            selector=k8s.V1LabelSelector(),
            template=k8s.V1PodTemplateSpec(spec=k8s.V1PodSpec(containers=specs)),
        ),
    )


class FakeApps:
    def __init__(self, deployments):
        self.deployments = {d.metadata.name: d for d in deployments}
        self.patches = []

    def read_namespaced_deployment(self, name, namespace):
        if name not in self.deployments:
            raise ApiException(status=404, reason="Not Found")
        return self.deployments[name]

    def patch_namespaced_deployment(self, name, namespace, body, dry_run=None):
        self.patches.append((name, dry_run, body))


@pytest.fixture
def backend():
    b = KubernetesBackend.__new__(KubernetesBackend)  # skip kubeconfig loading
    b.namespace = "cost-opt-sandbox"
    b.managed_label = "cost-opt.managed"
    b.apps = FakeApps([
        deployment("web-frontend", MANAGED, {"cpu": "250m", "memory": "256Mi"},
                   {"cpu": "500m", "memory": "512Mi"}),
        deployment("unmanaged", {}, {"memory": "256Mi"}, {"memory": "512Mi"}),
        deployment("sidecar", MANAGED, {"memory": "256Mi"}, {"memory": "512Mi"}, containers=2),
    ])
    return b


def test_set_requests_dry_runs_then_patches(backend):
    result = backend.apply(
        target="web-frontend", action="set_requests",
        params={"memory_mib": 128, "cpu_cores": 0.12}, metrics=make_metrics(),
    )
    assert result.ok, result.error
    assert "256Mi -> 128Mi" in result.detail and "250m -> 120m" in result.detail
    assert [dry for _, dry, _ in backend.apps.patches] == ["All", None]
    body = backend.apps.patches[-1][2]
    container = body["spec"]["template"]["spec"]["containers"][0]
    assert container == {
        "name": "web-frontend",
        "resources": {"requests": {"cpu": "120m", "memory": "128Mi"}},
    }


def test_scale_replicas(backend):
    result = backend.apply(target="web-frontend", action="scale_replicas",
                           params={"replicas": 3}, metrics=make_metrics())
    assert result.ok and result.detail == "replicas 2 -> 3"


@pytest.mark.parametrize(
    "target, action, params, metrics, reason",
    [
        ("web-frontend", "set_requests", {"memory_mib": 600}, make_metrics(),
         "would exceed the memory limit"),
        ("web-frontend", "set_requests", {"memory_mib": 16}, make_metrics(), "floor of 32"),
        ("web-frontend", "set_requests", {"memory_mib": 300},
         make_metrics(mem_max_bytes=900 * MIB), "minimum 1080"),
        ("web-frontend", "set_requests", {"cpu_cores": 0.02}, make_metrics(), "floor of 0.05"),
        ("web-frontend", "set_limits", {"memory_mib": 100}, make_metrics(), "hard floor of 128"),
        ("web-frontend", "set_limits", {"memory_mib": 200}, make_metrics(),
         "request 256Mi would exceed"),
        ("web-frontend", "scale_replicas", {"replicas": 0}, make_metrics(), "scaling to zero"),
        ("web-frontend", "scale_replicas", {"replicas": 3}, make_metrics(hpa_managed=True),
         "HorizontalPodAutoscaler"),
        ("web-frontend", "set_requests", {"memory_mib": 128},
         make_metrics(min_window="24h"), "at least 24h"),
        ("unmanaged", "set_requests", {"memory_mib": 128}, make_metrics(), "not labelled"),
        ("sidecar", "set_requests", {"memory_mib": 128}, make_metrics(), "2 containers"),
        ("ghost", "set_requests", {"memory_mib": 128}, make_metrics(), "does not exist"),
        ("web-frontend", "stop_container", {}, make_metrics(), "not supported"),
    ],
)
def test_guardrails_block_without_patching(backend, target, action, params, metrics, reason):
    result = backend.apply(target=target, action=action, params=params, metrics=metrics)
    assert not result.ok
    assert reason in result.error
    assert backend.apps.patches == []


def test_universal_actions_execute_nothing(backend):
    for action in ("no_action", "flag_for_review"):
        assert backend.apply(target="web-frontend", action=action, params={}, metrics=None).ok
    assert backend.apps.patches == []


def test_preflight_uses_observed_state(backend):
    observed = make_metrics(mem_request_bytes=256 * MIB, mem_limit_bytes=512 * MIB,
                            cpu_request_cores=0.25, cpu_limit_cores=0.5)
    assert backend.preflight(target="web-frontend", action="set_requests",
                             params={"memory_mib": 128}, metrics=observed) is None
    blocked = backend.preflight(target="web-frontend", action="set_requests",
                                params={"memory_mib": 600}, metrics=observed)
    assert "would exceed the memory limit" in blocked
    assert backend.apps.patches == []
