"""Tool definitions handed to the model, and the dispatcher that backs them.

Two tools only:

  search_policies  -- lets the model pull more corpus context than the seed
                      retrieval gave it, and is itself logged to the audit trail.
  propose_change   -- the structured-output channel. Calling it records a
                      proposal; it NEVER executes anything. Execution happens
                      after the loop, behind a typed human approval.

The set of actions `propose_change` offers depends on the backend: the model is
only ever shown actions the active backend can actually execute.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .audit import AuditLog
from .backends.base import WorkloadMetrics
from .rag import Chunk, PolicyStore


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


@dataclass
class Proposal:
    workload: str
    action: str
    params: dict[str, Any]
    reason: str
    policy_cited: list[str]
    confidence: str
    estimated_saving: str | None
    context: list[dict[str, Any]] = field(default_factory=list)
    decision_id: int | None = None

    @property
    def is_executable(self) -> bool:
        return self.action not in UNIVERSAL_ACTIONS

    def _resources(self) -> str:
        parts = []
        if self.params.get("memory_mib") is not None:
            parts.append(f"{self.params['memory_mib']} MiB memory")
        if self.params.get("cpu_cores") is not None:
            parts.append(f"{self.params['cpu_cores']} cores CPU")
        return ", ".join(parts)

    def describe(self) -> str:
        if self.action == "set_memory_limit":
            return f"set memory limit to {self.params.get('memory_mib')} MiB"
        if self.action == "set_cpu_limit":
            return f"set CPU limit to {self.params.get('cpu_cores')} cores"
        if self.action == "set_requests":
            return f"set requests to {self._resources()}"
        if self.action == "set_limits":
            return f"set limits to {self._resources()}"
        if self.action == "scale_replicas":
            return f"scale to {self.params.get('replicas')} replicas"
        return self.action.replace("_", " ")


class ToolContext:
    """Holds the per-run state the tool implementations need."""

    def __init__(
        self,
        *,
        store: PolicyStore,
        audit: AuditLog,
        run_id: str,
        metrics: list[WorkloadMetrics],
        seed_context: dict[str, list[Chunk]],
        backend_actions: tuple[str, ...],
        verbose: bool = True,
    ):
        self.store = store
        self.audit = audit
        self.run_id = run_id
        self.metrics = {m.name: m for m in metrics}
        self.targets = {m.name for m in metrics if m.managed}
        self.actions = available_actions(backend_actions)
        self.verbose = verbose
        self.proposals: list[Proposal] = []
        # Everything retrieved this run, per workload plus a shared pool from
        # tool-driven searches. Used to ground citations and to attach the
        # supporting context to each audit record.
        self.seed_context = seed_context
        self.retrieved: list[Chunk] = [c for chunks in seed_context.values() for c in chunks]

    # -- helpers ------------------------------------------------------------

    @property
    def known_sources(self) -> set[str]:
        return {c.source for c in self.retrieved}

    def _context_for(self, workload: str) -> list[dict[str, Any]]:
        chunks = list(self.seed_context.get(workload, [])) + self.retrieved
        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        for chunk in chunks:
            if chunk.id in seen:
                continue
            seen.add(chunk.id)
            out.append(chunk.to_dict())
        return out

    # -- tool implementations ----------------------------------------------

    def _search_policies(self, args: dict[str, Any]) -> tuple[str, bool]:
        query = str(args.get("query", "")).strip()
        if not query:
            return "Error: `query` must be a non-empty string.", True
        k = int(args.get("k") or self.store.settings.top_k)
        k = max(1, min(k, 8))

        chunks = self.store.search(query, k=k)
        self.retrieved.extend(chunks)
        self.audit.log_retrieval(
            self.run_id,
            origin="tool",
            query=query,
            results=[c.to_dict() for c in chunks],
        )
        if self.verbose:
            cites = ", ".join(c.citation for c in chunks)
            print(f"  [rag] search: {query!r}\n        -> {cites}")

        if not chunks:
            return "No matching passages found. Try a broader query.", False
        return (
            "\n\n".join(
                f"[{c.citation}] (distance {c.distance:.3f})\n{c.text}" for c in chunks
            ),
            False,
        )

    @staticmethod
    def _check_param(name: str, value: Any) -> str | None:
        """None if `value` is a usable value for parameter `name`."""
        if name == "memory_mib" or name == "replicas":
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                return f"`{name}` must be a positive integer"
        elif name == "cpu_cores":
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                return f"`{name}` must be a positive number"
        return None

    def _validate_proposal(self, args: dict[str, Any]) -> str | None:
        workload = str(args.get("workload", "")).strip()
        action = str(args.get("action", "")).strip()

        if workload not in self.metrics:
            known = ", ".join(sorted(self.targets))
            return f"Unknown workload `{workload}`. Workloads under review: {known}."
        if workload not in self.targets:
            return (
                f"`{workload}` is not a managed workload and is out of scope. "
                "Do not propose changes to platform or observability components."
            )
        if action not in self.actions:
            return f"Unknown action `{action}`. Valid actions: {', '.join(self.actions)}."

        # One decision per workload. A second call would give the approval
        # step two competing changes for the same target and split the audit
        # record for one judgement across two rows.
        already = next((p for p in self.proposals if p.workload == workload), None)
        if already is not None:
            return (
                f"A decision for `{workload}` is already recorded: "
                f"`{already.action}`. Record exactly one decision per workload "
                "-- if several changes look warranted, choose the single change "
                "with the larger impact and mention the others in `reason`."
            )

        spec = ACTION_SPECS[action]
        if spec.params:
            given = [p for p in spec.params if args.get(p) is not None]
            if spec.any_of and not given:
                return f"`{action}` needs at least one of: {', '.join(spec.params)}."
            if not spec.any_of and len(given) != len(spec.params):
                return f"`{action}` needs: {', '.join(spec.params)}."
            for p in given:
                problem = self._check_param(p, args[p])
                if problem:
                    return f"{problem} for {action}."

        citations = args.get("policy_cited") or []
        if not isinstance(citations, list) or not citations:
            return (
                "`policy_cited` must list at least one retrieved passage. Change "
                "management requires every decision to name its supporting document; "
                "call search_policies first if you have nothing to cite."
            )
        sources = self.known_sources
        if sources and not any(
            any(src in str(cite) for src in sources) for cite in citations
        ):
            return (
                "None of the citations match a document retrieved in this run. Cite "
                "passages returned by search_policies, using their exact citation "
                f"address. Documents retrieved so far: {', '.join(sorted(sources))}."
            )
        if not str(args.get("reason", "")).strip():
            return "`reason` must explain the decision; it cannot be empty."
        return None

    def _propose_change(self, args: dict[str, Any]) -> tuple[str, bool]:
        problem = self._validate_proposal(args)
        if problem:
            return f"Proposal rejected: {problem}", True

        workload = args["workload"]
        action = args["action"]
        params: dict[str, Any] = {}
        for p in ACTION_SPECS[action].params:
            if args.get(p) is not None:
                params[p] = float(args[p]) if p == "cpu_cores" else int(args[p])

        context = self._context_for(workload)
        proposal = Proposal(
            workload=workload,
            action=action,
            params=params,
            reason=str(args["reason"]).strip(),
            policy_cited=[str(c) for c in args["policy_cited"]],
            confidence=str(args.get("confidence", "medium")),
            estimated_saving=args.get("estimated_saving"),
            context=context,
        )
        proposal.decision_id = self.audit.log_proposal(
            self.run_id,
            container=workload,
            action=action,
            params=params,
            reason=proposal.reason,
            policy_cited=proposal.policy_cited,
            confidence=proposal.confidence,
            estimated_saving=proposal.estimated_saving,
            context=context,
        )
        self.proposals.append(proposal)
        if self.verbose:
            print(f"  [proposal] {workload}: {proposal.describe()} ({proposal.confidence})")

        return (
            json.dumps(
                {
                    "recorded": True,
                    "decision_id": proposal.decision_id,
                    "workload": workload,
                    "action": action,
                    "status": "pending_human_approval",
                    "note": "Nothing has been executed. A human will approve or reject this.",
                }
            ),
            False,
        )

    # -- dispatch -----------------------------------------------------------

    def dispatch(self, name: str, args: dict[str, Any]) -> tuple[str, bool]:
        """Returns (content, is_error) for a tool_result block."""
        if name == "search_policies":
            return self._search_policies(args)
        if name == "propose_change":
            return self._propose_change(args)
        return f"Unknown tool `{name}`.", True
