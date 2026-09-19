from types import SimpleNamespace

from agent.providers import _history_message, max_tokens_for, to_openai_tools
from agent.config import Settings
from agent.tools import build_tool_defs


def test_tool_schemas_translate_to_openai_shape():
    tools = to_openai_tools(build_tool_defs(("set_requests",)))
    assert [t["function"]["name"] for t in tools] == ["search_policies", "propose_change"]
    assert all(t["type"] == "function" and "parameters" in t["function"] for t in tools)


def test_history_drops_reasoning_but_keeps_tool_calls():
    call = SimpleNamespace(id="c1", function=SimpleNamespace(name="propose_change",
                                                              arguments='{"a": 1}'))
    message = SimpleNamespace(content=None, tool_calls=[call], reasoning="long chain...")
    out = _history_message(message)
    assert "reasoning" not in out
    assert out["content"] == ""
    assert out["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'


def test_output_budget_is_per_provider_unless_set():
    assert max_tokens_for(Settings(provider="groq", max_tokens=0)) == 3072
    assert max_tokens_for(Settings(provider="anthropic", max_tokens=0)) == 16000
    assert max_tokens_for(Settings(provider="groq", max_tokens=1000)) == 1000
