"""Execution backends. Selected with BACKEND=docker|kubernetes."""

from __future__ import annotations

from ..config import Settings
from .base import Backend

BACKENDS = ("docker", "kubernetes")


def build_backend(settings: Settings, *, allow_stop: bool = False) -> Backend:
    if settings.backend == "docker":
        from .docker import DockerBackend

        return DockerBackend(settings, allow_stop=allow_stop)
    if settings.backend == "kubernetes":
        from .kubernetes import KubernetesBackend

        return KubernetesBackend(settings)
    raise ValueError(
        f"Unknown backend `{settings.backend}`. Choose one of: {', '.join(BACKENDS)}."
    )
