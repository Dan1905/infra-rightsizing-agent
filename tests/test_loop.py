"""The reasoning loop driven by a scripted provider -- no model involved."""

from agent.config import Settings
from agent.llm import run_analysis
from agent.providers import LLMResponse, ToolCall

CITE = ["workload-profiles.md > web-frontend"]


class ScriptedProvider:
    name, model = "scripted", "none"

    def __init__(self, turns):
        self.turns = list(turns)
        self.results = []
        self.prompts = []

    def initial_messages(self, system, user_text):
        return [{"role": "user", "content": user_text}]

    def call(self, messages, system):
        self.prompts.append(messages[-1])
        calls = self.turns.pop(0) if self.turns else []
        return LLMResponse("", calls, "tool_calls" if calls else "stop", {"role": "assistant"})

    def tool_result_messages(self, results):
        self.results += [(c.args.get("workload"), is_error) for c, _, is_error in results]
        return [{"role": "tool"}]


def decide(workload, action="no_action", **extra):
    return ToolCall(f"id-{workload}-{action}", "propose_change", {
        "workload": workload, "action": action, "reason": "r",
        "policy_cited": CITE, "confidence": "high", **extra,
    })


def run(provider, store, audit, workloads, **kw):
    run_id = audit.start_run(model="t", lookback="1h", prometheus_url="x")
    return run_analysis(
        Settings(max_turns=6), metrics=workloads, store=store, audit=audit,
        run_id=run_id, backend_actions=("set_requests",), provider=provider,
        verbose=False, **kw,
    )


def test_every_workload_decided(store, audit, workloads):
    provider = ScriptedProvider([[
        decide("web-frontend", "set_requests", memory_mib=128),
        decide("batch-worker"), decide("payment-service", "flag_for_review"),
    ]])
    proposals, _, transcript = run(provider, store, audit, workloads)
    assert {p.workload for p in proposals} == {"web-frontend", "batch-worker", "payment-service"}
    assert transcript[0]["role"] == "user"


def test_missing_decisions_are_asked_for_again(store, audit, workloads):
    provider = ScriptedProvider([
        [decide("web-frontend")],
        [],                                   # stops early
        [decide("batch-worker"), decide("payment-service")],
    ])
    proposals, _, _ = run(provider, store, audit, workloads)
    assert len(proposals) == 3
    nudge = provider.prompts[2]
    assert "batch-worker" in nudge["content"] and "payment-service" in nudge["content"]


def test_nudging_is_bounded(store, audit, workloads):
    provider = ScriptedProvider([[decide("web-frontend")]])  # then never again
    proposals, _, _ = run(provider, store, audit, workloads)
    assert len(proposals) == 1
    assert len(provider.prompts) == 4  # first call + one after results + two nudges


def test_malformed_arguments_go_back_as_errors(store, audit, workloads):
    bad = ToolCall("bad", "propose_change", {"__parse_error__": "{not json"})
    provider = ScriptedProvider([[bad], [decide("web-frontend"), decide("batch-worker"),
                                         decide("payment-service")]])
    proposals, _, _ = run(provider, store, audit, workloads)
    assert provider.results[0] == (None, True)
    assert len(proposals) == 3


def test_seed_retrieval_runs_per_managed_workload(store, audit, workloads):
    run(ScriptedProvider([]), store, audit, workloads)
    # metrics + constraint + platform query for each of 3 managed Deployments
    assert len(store.queries) == 9
    assert not any("prometheus" in q for q in store.queries)
