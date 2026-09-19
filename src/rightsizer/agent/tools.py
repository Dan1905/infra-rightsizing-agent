"""The tools offered to the model, and the actions `propose_change` can carry.

Two tools only:

  search_policies  -- lets the model pull more corpus context than the seed
                      retrieval gave it, and is itself logged to the audit trail.
  propose_change   -- the structured-output channel. Calling it records a
                      proposal; it NEVER executes anything. Execution happens
                      after the loop, behind a typed human approval.

The set of actions `propose_change` offers depends on the backend: the model is
only ever shown actions the active backend can actually execute. What happens
when a tool is called lives in proposals.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any



@dataclass(frozen=True)
class ActionSpec:
    description: str
    # Parameters the action needs. With `any_of`, at least one must be given;
    # otherwise all of them are required.
    params: tuple[str, ...] = ()
    any_of: bool = False


ACTION_SPECS: dict[str, ActionSpec] = {
    # Docker
    "set_memory_limit": ActionSpec(
        "Lower (or raise) the container's memory limit. Needs memory_mib.",
        ("memory_mib",),
    ),
    "set_cpu_limit": ActionSpec(
        "Lower (or raise) the container's CPU limit. Needs cpu_cores.",
        ("cpu_cores",),
    ),
    "stop_container": ActionSpec("Stop a container that is confirmed abandoned."),
    # Kubernetes
    "set_requests": ActionSpec(
        "Change the per-pod resource REQUESTS -- what the scheduler reserves, and "
        "therefore what drives node count and cost. Give memory_mib and/or cpu_cores.",
        ("memory_mib", "cpu_cores"),
        any_of=True,
    ),
    "set_limits": ActionSpec(
        "Change the per-pod resource LIMITS -- the ceiling a pod is killed or "
        "throttled at. Give memory_mib and/or cpu_cores.",
        ("memory_mib", "cpu_cores"),
        any_of=True,
    ),
    "scale_replicas": ActionSpec(
        "Change the Deployment's replica count. Needs replicas.",
        ("replicas",),
    ),
    # Every backend
    "flag_for_review": ActionSpec("Take no automated action; route to the owning team."),
    "no_action": ActionSpec(
        "The workload is correctly sized, or the evidence is insufficient."
    ),
}

UNIVERSAL_ACTIONS = ("flag_for_review", "no_action")


def available_actions(backend_actions: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(backend_actions) + UNIVERSAL_ACTIONS


def build_tool_defs(backend_actions: tuple[str, ...]) -> list[dict[str, Any]]:
    """The tool schemas for one backend. Provider-neutral (Anthropic shape);
    providers.py translates for OpenAI-compatible endpoints."""
    actions = available_actions(backend_actions)
    return [
        {
            "name": "search_policies",
            "description": (
                "Semantic search over the organisation's policy documents, service "
                "runbooks, workload profiles and incident postmortems. Use this "
                "whenever a metric reading might be explained or constrained by a "
                "document -- especially before proposing any change to a workload "
                "you have not yet retrieved specific context for. Returns the "
                "top-k matching passages with their citation addresses."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Natural-language query. Describe the workload and the "
                            "signal you are trying to interpret, e.g. 'batch-worker "
                            "idle during the day, nightly job memory requirements'."
                        ),
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of passages to return (1-8). Defaults to 4.",
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "propose_change",
            "description": (
                "Record exactly one decision for exactly one workload. Call this "
                "once per workload you were asked to assess, including workloads "
                "you decide to leave alone (use action `no_action`). This records a "
                "proposal only -- nothing is executed until a human approves it."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "workload": {
                        "type": "string",
                        "description": "Workload name, exactly as given in the metrics.",
                    },
                    "action": {
                        "type": "string",
                        "enum": list(actions),
                        "description": "\n".join(
                            f"{a}: {ACTION_SPECS[a].description}" for a in actions
                        ),
                    },
                    # Optional parameters are nullable: models routinely send an
                    # explicit null rather than omitting a field, and providers
                    # that validate tool arguments server-side reject that
                    # against a bare "integer" with a 400.
                    "memory_mib": {
                        "type": ["integer", "null"],
                        "description": "Memory in MiB for the chosen action, null if unused.",
                    },
                    "cpu_cores": {
                        "type": ["number", "null"],
                        "description": "CPU in cores for the chosen action, null if unused.",
                    },
                    **(
                        {
                            "replicas": {
                                "type": ["integer", "null"],
                                "description": "Replica count for scale_replicas, null otherwise.",
                            }
                        }
                        if "scale_replicas" in actions
                        else {}
                    ),
                    "reason": {
                        "type": "string",
                        "description": (
                            "Why this decision follows from the metrics AND the "
                            "retrieved documents. Name the specific numbers you "
                            "relied on and what the document said about them. "
                            "Write for an on-call engineer reviewing at 3am."
                        ),
                    },
                    "policy_cited": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Citation addresses of the passages that support this "
                            "decision, copied from search results, e.g. "
                            "'payment-service-runbook.md > Binding constraints'. "
                            "At least one is required for every decision."
                        ),
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                    "estimated_saving": {
                        "type": ["string", "null"],
                        "description": (
                            "Short human-readable estimate, e.g. '768 MiB memory "
                            "reclaimed'. Null when nothing is saved."
                        ),
                    },
                },
                "required": ["workload", "action", "reason", "policy_cited", "confidence"],
            },
        },
    ]
