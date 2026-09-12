"""Tool definitions handed to Claude, and the dispatcher that backs them.

Two tools only:

  search_policies  -- lets the model pull more corpus context than the seed
                      retrieval gave it, and is itself logged to the audit trail.
  propose_change   -- the structured-output channel. Calling it records a
                      proposal; it NEVER executes anything. Execution happens
                      after the loop, behind a typed human approval.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .audit import AuditLog
from .metrics import ContainerMetrics
from .rag import Chunk, PolicyStore

ACTIONS = {
    "set_memory_limit": "Lower (or raise) the container's memory limit.",
    "set_cpu_limit": "Lower (or raise) the container's CPU limit, in cores.",
    "stop_container": "Stop a container that is confirmed abandoned.",
    "flag_for_review": "Take no automated action; route to the owning team.",
    "no_action": "The container is correctly sized, or the evidence is insufficient.",
}

TOOL_DEFS: list[dict[str, Any]] = [
    {
        "name": "search_policies",
        "description": (
            "Semantic search over the organisation's policy documents, service "
            "runbooks, workload profiles and incident postmortems. Use this "
            "whenever a metric reading might be explained or constrained by a "
            "document -- especially before proposing any change to a container "
            "you have not yet retrieved specific context for. Returns the "
            "top-k matching passages with their citation addresses."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Natural-language query. Describe the container and the "
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
            "Record exactly one decision for exactly one container. Call this "
            "once per container you were asked to assess, including containers "
            "you decide to leave alone (use action `no_action`). This records a "
            "proposal only -- nothing is executed until a human approves it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "container": {
                    "type": "string",
                    "description": "Container name, exactly as given in the metrics.",
                },
                "action": {
                    "type": "string",
                    "enum": list(ACTIONS),
                    "description": "\n".join(f"{k}: {v}" for k, v in ACTIONS.items()),
                },
                "memory_mib": {
                    # Nullable: models routinely send an explicit null for
                    # optional fields rather than omitting them, and providers
                    # that validate tool arguments server-side reject that
                    # against a bare "integer" with a 400.
                    "type": ["integer", "null"],
                    "description": "New memory limit in MiB. Required for set_memory_limit, null otherwise.",
                },
                "cpu_cores": {
                    "type": ["number", "null"],
                    "description": "New CPU limit in cores. Required for set_cpu_limit, null otherwise.",
                },
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
                    "description": "Short human-readable estimate, e.g. '768 MiB memory reclaimed'. Null when nothing is saved.",
                },
            },
            "required": ["container", "action", "reason", "policy_cited", "confidence"],
        },
    },
]


@dataclass
class Proposal:
    container: str
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
        return self.action in ("set_memory_limit", "set_cpu_limit", "stop_container")

    def describe(self) -> str:
        if self.action == "set_memory_limit":
            return f"set memory limit to {self.params.get('memory_mib')} MiB"
        if self.action == "set_cpu_limit":
            return f"set CPU limit to {self.params.get('cpu_cores')} cores"
        return self.action.replace("_", " ")


class ToolContext:
    """Holds the per-run state the tool implementations need."""

    def __init__(
        self,
        *,
        store: PolicyStore,
        audit: AuditLog,
        run_id: str,
        metrics: list[ContainerMetrics],
        seed_context: dict[str, list[Chunk]],
        verbose: bool = True,
    ):
        self.store = store
        self.audit = audit
        self.run_id = run_id
        self.metrics = {m.name: m for m in metrics}
        self.targets = {m.name for m in metrics if m.managed}
        self.verbose = verbose
        self.proposals: list[Proposal] = []
        # Everything retrieved this run, per container plus a shared pool from
        # tool-driven searches. Used to ground citations and to attach the
        # supporting context to each audit record.
        self.seed_context = seed_context
        self.retrieved: list[Chunk] = [c for chunks in seed_context.values() for c in chunks]

    # -- helpers ------------------------------------------------------------

    @property
    def known_sources(self) -> set[str]:
        return {c.source for c in self.retrieved}

    def _context_for(self, container: str) -> list[dict[str, Any]]:
        chunks = list(self.seed_context.get(container, [])) + self.retrieved
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

    def _validate_proposal(self, args: dict[str, Any]) -> str | None:
        container = str(args.get("container", "")).strip()
        action = str(args.get("action", "")).strip()

        if container not in self.metrics:
            known = ", ".join(sorted(self.targets))
            return f"Unknown container `{container}`. Containers under review: {known}."
        if container not in self.targets:
            return (
                f"`{container}` is not a managed workload and is out of scope. "
                "Do not propose changes to platform or observability containers."
            )
        if action not in ACTIONS:
            return f"Unknown action `{action}`. Valid actions: {', '.join(ACTIONS)}."

        # One decision per container. A second call would give the approval
        # step two competing changes for the same target and split the audit
        # record for one judgement across two rows.
        already = next((p for p in self.proposals if p.container == container), None)
        if already is not None:
            return (
                f"A decision for `{container}` is already recorded: "
                f"`{already.action}`. Record exactly one decision per container "
                "-- if both CPU and memory look oversized, choose the single "
                "change with the larger impact and mention the other in `reason`."
            )

        if action == "set_memory_limit":
            value = args.get("memory_mib")
            if not isinstance(value, int) or value <= 0:
                return "`memory_mib` must be a positive integer for set_memory_limit."
        if action == "set_cpu_limit":
            value = args.get("cpu_cores")
            if not isinstance(value, (int, float)) or value <= 0:
                return "`cpu_cores` must be a positive number for set_cpu_limit."

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

        container = args["container"]
        action = args["action"]
        params: dict[str, Any] = {}
        if action == "set_memory_limit":
            params["memory_mib"] = int(args["memory_mib"])
        elif action == "set_cpu_limit":
            params["cpu_cores"] = float(args["cpu_cores"])

        context = self._context_for(container)
        proposal = Proposal(
            container=container,
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
            container=container,
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
            print(f"  [proposal] {container}: {proposal.describe()} ({proposal.confidence})")

        return (
            json.dumps(
                {
                    "recorded": True,
                    "decision_id": proposal.decision_id,
                    "container": container,
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
