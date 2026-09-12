"""Provider abstraction for the tool-calling loop.

The agent's logic -- retrieve, reason, propose, approve -- is provider-agnostic.
Only two things differ between vendors: the wire format for tool calls and the
shape of the conversation history. Each provider below owns exactly that.

Defaults target Groq's free tier (gpt-oss-120b), which speaks the OpenAI
chat-completions dialect. Anthropic and any other OpenAI-compatible endpoint
(xAI, Together, OpenRouter, a local vLLM) are selected with LLM_PROVIDER.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

from .config import Settings

# provider -> (base_url, default model, env var holding the key)
PRESETS: dict[str, tuple[str | None, str, str]] = {
    "groq": (
        "https://api.groq.com/openai/v1",
        "openai/gpt-oss-120b",
        "GROQ_API_KEY",
    ),
    "anthropic": (None, "claude-opus-5", "ANTHROPIC_API_KEY"),
    # Generic OpenAI-compatible endpoint: set LLM_BASE_URL and LLM_MODEL.
    # e.g. xAI -> https://api.x.ai/v1, Together -> https://api.together.xyz/v1
    "openai": ("https://api.openai.com/v1", "gpt-4o-mini", "OPENAI_API_KEY"),
}


class ProviderError(RuntimeError):
    """Configuration or transport problem talking to the model provider."""


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = ""
    assistant_message: Any = None  # appended verbatim to the history


class Provider(Protocol):
    name: str
    model: str

    def initial_messages(self, system: str, user_text: str) -> list[Any]: ...
    def call(self, messages: list[Any], system: str) -> LLMResponse: ...
    def tool_result_messages(
        self, results: list[tuple[ToolCall, str, bool]]
    ) -> list[Any]: ...


# --------------------------------------------------------------------------- #
# Tool schema translation
# --------------------------------------------------------------------------- #


def to_openai_tools(tool_defs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Anthropic-style tool defs -> OpenAI function-calling defs."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in tool_defs
    ]


# --------------------------------------------------------------------------- #
# OpenAI-compatible (Groq, xAI, OpenAI, vLLM, ...)
# --------------------------------------------------------------------------- #


class OpenAICompatibleProvider:
    """Chat-completions dialect: system as messages[0], tool results as
    separate `role: tool` messages keyed by tool_call_id."""

    def __init__(self, settings: Settings, tool_defs: list[dict[str, Any]]):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ProviderError(
                "The `openai` package is required for this provider: pip install openai"
            ) from exc

        base_url, default_model, key_env = PRESETS[settings.provider]
        api_key = settings.api_key or os.environ.get(key_env) or os.environ.get("LLM_API_KEY")
        if not api_key:
            raise ProviderError(
                f"No API key for provider `{settings.provider}`. "
                f"Set {key_env} (or LLM_API_KEY)."
            )

        self.name = settings.provider
        self.model = settings.model or default_model
        self.settings = settings
        self.tools = to_openai_tools(tool_defs)
        self._client = OpenAI(
            api_key=api_key,
            base_url=settings.base_url or base_url,
            max_retries=settings.max_retries,
        )

    def initial_messages(self, system: str, user_text: str) -> list[Any]:
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ]

    def call(self, messages: list[Any], system: str) -> LLMResponse:
        import openai

        try:
            completion = self._client.chat.completions.create(
                model=self.model,
                max_tokens=self.settings.max_tokens,
                temperature=self.settings.temperature,
                tools=self.tools,
                tool_choice="auto",
                messages=messages,
            )
        except openai.AuthenticationError as exc:
            _, _, key_env = PRESETS[self.name]
            raise ProviderError(
                f"{self.name} rejected the credentials ({exc.message}). Check {key_env}."
            ) from exc
        except openai.NotFoundError as exc:
            raise ProviderError(
                f"Model `{self.model}` was not found on {self.name}. "
                "Set LLM_MODEL to a model the endpoint serves."
            ) from exc
        except openai.RateLimitError as exc:
            # Free tiers have tight per-minute token budgets; the SDK has
            # already retried max_retries times by the time this surfaces.
            raise ProviderError(
                f"{self.name} rate limit reached after {self.settings.max_retries} "
                f"retries ({exc.message}). Try a shorter --lookback, a lower "
                "RAG_TOP_K, or wait a minute."
            ) from exc
        except openai.APIStatusError as exc:
            raise ProviderError(f"{self.name} returned {exc.status_code}: {exc.message}") from exc
        except openai.APIConnectionError as exc:
            raise ProviderError(f"Could not reach {self.name}: {exc}") from exc

        if not completion.choices:
            raise ProviderError(f"{self.name} returned no choices.")
        choice = completion.choices[0]
        message = choice.message

        calls: list[ToolCall] = []
        for call in message.tool_calls or []:
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                # Smaller models occasionally emit malformed argument JSON.
                # Surface it as a tool error so the model can correct itself
                # rather than crashing the run.
                args = {"__parse_error__": call.function.arguments}
            calls.append(ToolCall(id=call.id, name=call.function.name, args=args))

        return LLMResponse(
            text=(message.content or "").strip(),
            tool_calls=calls,
            stop_reason=choice.finish_reason or "",
            assistant_message=message.model_dump(exclude_none=True),
        )

    def tool_result_messages(
        self, results: list[tuple[ToolCall, str, bool]]
    ) -> list[Any]:
        return [
            {
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.name,
                "content": f"ERROR: {content}" if is_error else content,
            }
            for call, content, is_error in results
        ]


# --------------------------------------------------------------------------- #
# Anthropic
# --------------------------------------------------------------------------- #


class AnthropicProvider:
    """Messages dialect: system is a top-level parameter, and every tool result
    for one assistant turn goes back in a single user message."""

    def __init__(self, settings: Settings, tool_defs: list[dict[str, Any]]):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise ProviderError(
                "The `anthropic` package is required for this provider: pip install anthropic"
            ) from exc

        self._anthropic = anthropic
        self.name = "anthropic"
        self.model = settings.model or PRESETS["anthropic"][1]
        self.settings = settings
        self.tools = tool_defs
        kwargs: dict[str, Any] = {"max_retries": settings.max_retries}
        if settings.api_key:
            kwargs["api_key"] = settings.api_key
        if settings.base_url:
            kwargs["base_url"] = settings.base_url
        self._client = anthropic.Anthropic(**kwargs)

    def initial_messages(self, system: str, user_text: str) -> list[Any]:
        return [{"role": "user", "content": user_text}]

    def call(self, messages: list[Any], system: str) -> LLMResponse:
        anthropic = self._anthropic
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.settings.max_tokens,
                system=system,
                thinking={"type": "adaptive"},
                output_config={"effort": self.settings.effort},
                tools=self.tools,
                messages=messages,
            )
        except TypeError as exc:
            # The SDK raises TypeError, not an APIError, when no credential can
            # be resolved at all.
            if "authentication method" not in str(exc):
                raise
            raise ProviderError(
                "No Anthropic credentials found. Set ANTHROPIC_API_KEY or run `ant auth login`."
            ) from exc
        except anthropic.AuthenticationError as exc:
            raise ProviderError(f"Anthropic rejected the credentials: {exc.message}") from exc
        except anthropic.NotFoundError as exc:
            raise ProviderError(f"Model `{self.model}` not found: {exc.message}") from exc
        except anthropic.RateLimitError as exc:
            raise ProviderError(
                f"Anthropic rate limit reached after {self.settings.max_retries} "
                f"retries: {exc.message}"
            ) from exc
        except anthropic.APIStatusError as exc:
            raise ProviderError(f"Anthropic returned {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderError(f"Could not reach the Anthropic API: {exc}") from exc

        if response.stop_reason == "refusal":
            detail = getattr(response.stop_details, "explanation", None)
            raise ProviderError(f"Model declined the request: {detail}")

        calls = [
            ToolCall(
                id=b.id,
                name=b.name,
                args=b.input if isinstance(b.input, dict) else json.loads(b.input),
            )
            for b in response.content
            if b.type == "tool_use"
        ]
        text = "\n".join(b.text for b in response.content if b.type == "text").strip()
        return LLMResponse(
            text=text,
            tool_calls=calls,
            stop_reason=response.stop_reason or "",
            assistant_message={"role": "assistant", "content": response.content},
        )

    def tool_result_messages(
        self, results: list[tuple[ToolCall, str, bool]]
    ) -> list[Any]:
        blocks = [
            {
                "type": "tool_result",
                "tool_use_id": call.id,
                "content": content,
                **({"is_error": True} if is_error else {}),
            }
            for call, content, is_error in results
        ]
        return [{"role": "user", "content": blocks}]


def build_provider(settings: Settings, tool_defs: list[dict[str, Any]]) -> Provider:
    if settings.provider not in PRESETS:
        raise ProviderError(
            f"Unknown provider `{settings.provider}`. "
            f"Choose one of: {', '.join(PRESETS)}."
        )
    if settings.provider == "anthropic":
        return AnthropicProvider(settings, tool_defs)
    return OpenAICompatibleProvider(settings, tool_defs)
