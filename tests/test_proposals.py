"""propose_change validation: what the model is told when a proposal is wrong."""

import json

import pytest

from agent.tools import ToolContext, build_tool_defs

K8S = ("set_requests", "set_limits", "scale_replicas")
DOCKER = ("set_memory_limit", "set_cpu_limit", "stop_container")
CITE = ["workload-profiles.md > web-frontend"]


@pytest.fixture
def ctx(store, audit, workloads, corpus):
    run_id = audit.start_run(model="test", lookback="1h", prometheus_url="x")
    return ToolContext(
        store=store, audit=audit, run_id=run_id, metrics=workloads,
        seed_context={"web-frontend": corpus}, backend_actions=K8S,
    )


def propose(ctx, **args):
    args.setdefault("reason", "because")
    args.setdefault("policy_cited", CITE)
    args.setdefault("confidence", "high")
    return ctx.dispatch("propose_change", args)


def test_valid_proposal_is_recorded_not_executed(ctx, audit):
    content, is_error = propose(ctx, workload="web-frontend", action="set_requests",
                                memory_mib=128)
    assert not is_error
    assert json.loads(content)["status"] == "pending_human_approval"
    [row] = audit.recent_decisions()
    assert row["container"] == "web-frontend" and row["executed"] == 0
    assert ctx.proposals[0].describe() == "set requests to 128 MiB memory"


@pytest.mark.parametrize(
    "args, message",
    [
        ({"workload": "ghost", "action": "no_action"}, "Unknown workload"),
        ({"workload": "prometheus", "action": "no_action"}, "out of scope"),
        ({"workload": "web-frontend", "action": "set_memory_limit", "memory_mib": 128},
         "Unknown action"),
        ({"workload": "web-frontend", "action": "set_requests"}, "at least one of"),
        ({"workload": "web-frontend", "action": "scale_replicas", "replicas": None}, "needs"),
        ({"workload": "web-frontend", "action": "set_requests", "memory_mib": -5},
         "positive integer"),
        ({"workload": "web-frontend", "action": "set_requests", "memory_mib": True},
         "positive integer"),
        ({"workload": "web-frontend", "action": "no_action", "policy_cited": []},
         "at least one retrieved passage"),
        ({"workload": "web-frontend", "action": "no_action",
          "policy_cited": ["made-up.md > Nowhere"]}, "None of the citations"),
        ({"workload": "web-frontend", "action": "no_action", "reason": "  "},
         "cannot be empty"),
    ],
)
def test_rejections_explain_themselves(ctx, args, message):
    content, is_error = propose(ctx, **args)
    assert is_error
    assert message in content
    assert ctx.proposals == []


def test_one_decision_per_workload(ctx):
    assert not propose(ctx, workload="web-frontend", action="no_action")[1]
    content, is_error = propose(ctx, workload="web-frontend", action="flag_for_review")
    assert is_error and "already recorded" in content


def test_preflight_rejection_is_returned_to_the_model(ctx):
    ctx.preflight = lambda **kw: "64 MiB is below the hard floor of 128 MiB"
    content, is_error = propose(ctx, workload="web-frontend", action="set_limits",
                                memory_mib=64)
    assert is_error and "blocked by a guardrail" in content and "128 MiB" in content


def test_preflight_skipped_for_decisions_that_change_nothing(ctx):
    ctx.preflight = lambda **kw: "should not be called"
    assert not propose(ctx, workload="web-frontend", action="no_action")[1]


def test_schema_offers_only_backend_actions():
    for actions, absent in ((K8S, "set_memory_limit"), (DOCKER, "set_requests")):
        props = build_tool_defs(actions)[1]["input_schema"]["properties"]
        assert absent not in props["action"]["enum"]
        assert {"flag_for_review", "no_action"} <= set(props["action"]["enum"])
    assert "replicas" in build_tool_defs(K8S)[1]["input_schema"]["properties"]
    assert "replicas" not in build_tool_defs(DOCKER)[1]["input_schema"]["properties"]


def test_search_is_logged_and_widens_citable_sources(ctx, audit, store):
    content, is_error = ctx.dispatch("search_policies", {"query": "floors", "k": 2})
    assert not is_error and "[workload-profiles.md > web-frontend]" in content
    assert ctx.dispatch("search_policies", {"query": "  "})[1]
    assert store.queries == ["floors"]
