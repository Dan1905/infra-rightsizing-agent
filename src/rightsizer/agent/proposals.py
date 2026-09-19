"""What happens when the model calls a tool.

`propose_change` is where a model decision is checked before anyone sees it:
the workload must be in scope, the action offered by this backend, its
parameters present and sane, its citations drawn from passages actually
retrieved this run, and the change must pass the backend's guardrails. Every
rejection goes back to the model as a tool error it can act on. An accepted
proposal is written to the audit log -- and nothing more.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from ..audit import AuditLog
from ..backends.base import WorkloadMetrics
from ..retrieval.store import Chunk, PolicyStore
from .tools import ACTION_SPECS, UNIVERSAL_ACTIONS, available_actions


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
        preflight: Callable[..., str | None] | None = None,
        verbose: bool = True,
    ):
        self.store = store
        self.audit = audit
        self.run_id = run_id
        self.metrics = {m.name: m for m in metrics}
        self.targets = {m.name for m in metrics if m.managed}
        self.actions = available_actions(backend_actions)
        self.preflight = preflight
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
        def body(c: Chunk) -> str:
            # Stored text starts with its own address line; the header has it.
            return c.text.split("\n", 1)[1] if c.text.startswith("[") else c.text

        return (
            "\n\n".join(
                f"[{c.citation}] (distance {c.distance:.3f})\n{body(c)}" for c in chunks
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

        # Run the backend's guardrails now, against the observed state, so a
        # proposal that would be blocked at execution is corrected here rather
        # than discovered by the human after they approve it.
        if self.preflight is not None and action not in UNIVERSAL_ACTIONS:
            params = {p: args[p] for p in spec.params if args.get(p) is not None}
            blocked = self.preflight(
                target=workload, action=action, params=params,
                metrics=self.metrics.get(workload),
            )
            if blocked:
                return (
                    f"this change would be blocked by a guardrail at execution: "
                    f"{blocked}. Propose a value that satisfies it, or a different action."
                )

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
