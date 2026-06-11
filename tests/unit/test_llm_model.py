"""Unit tests for the OpenRouter function-calling client (helprers/llm_model.py).

These tests target ``LLMModel.call_tool`` itself, so the underlying OpenAI-compatible
client (``chat.completions.create``) is mocked — NOT the conftest ``mock_llm`` seam.
No live API calls are made.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from helprers.llm_model import LLMModel, ProviderResponseError
from schemas import EXTRACT_TIMEOUT_S

_TOOL_SCHEMA = {
    "name": "extract_cv_facts",
    "description": "Extract structured facts.",
    "parameters": {
        "type": "object",
        "required": ["personal_info"],
        "properties": {"personal_info": {"type": "object"}},
    },
}


def _fake_completion(arguments: dict) -> SimpleNamespace:
    """Build an OpenAI-shaped chat-completion carrying a single forced tool call."""
    tool_call = SimpleNamespace(
        function=SimpleNamespace(name="extract_cv_facts", arguments=json.dumps(arguments))
    )
    message = SimpleNamespace(tool_calls=[tool_call])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


async def test_call_tool_returns_parsed_arguments_dict() -> None:
    model = LLMModel(api_key="sk-or-test-dummy")
    expected = {"personal_info": {"name": "Ada", "location": "London", "email": "a@b.co"}}
    model.client = MagicMock(name="openai_client")
    model.client.chat.completions.create = AsyncMock()
    model.client.chat.completions.create.return_value = _fake_completion(expected)

    result = await model.call_tool("system text", "user text", _TOOL_SCHEMA)

    assert result == expected


async def test_call_tool_sends_system_user_and_tool_schema() -> None:
    model = LLMModel(api_key="sk-or-test-dummy")
    model.client = MagicMock(name="openai_client")
    model.client.chat.completions.create = AsyncMock()
    model.client.chat.completions.create.return_value = _fake_completion({"personal_info": {}})

    await model.call_tool("SYSTEM_PROMPT", "USER_PROMPT", _TOOL_SCHEMA)

    _, kwargs = model.client.chat.completions.create.call_args

    # Messages carry both the system and the user prompt.
    messages = kwargs["messages"]
    roles = {m["role"]: m["content"] for m in messages}
    assert roles["system"] == "SYSTEM_PROMPT"
    assert roles["user"] == "USER_PROMPT"

    # The tool schema is forwarded under the function-calling envelope.
    tools = kwargs["tools"]
    assert any(
        tool["type"] == "function" and tool["function"]["name"] == "extract_cv_facts"
        for tool in tools
    )

    # The model is forced to call this specific tool.
    tool_choice = kwargs["tool_choice"]
    assert tool_choice["type"] == "function"
    assert tool_choice["function"]["name"] == "extract_cv_facts"


async def test_call_tool_applies_extract_timeout() -> None:
    model = LLMModel(api_key="sk-or-test-dummy")
    model.client = MagicMock(name="openai_client")
    model.client.chat.completions.create = AsyncMock()
    model.client.chat.completions.create.return_value = _fake_completion({"personal_info": {}})

    await model.call_tool("system", "user", _TOOL_SCHEMA)

    _, kwargs = model.client.chat.completions.create.call_args
    assert kwargs["timeout"] == EXTRACT_TIMEOUT_S


async def test_call_tool_raises_provider_response_error_when_no_tool_call() -> None:
    model = LLMModel(api_key="sk-or-test-dummy")
    model.client = MagicMock(name="openai_client")
    model.client.chat.completions.create = AsyncMock()
    model.client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=None))]
    )

    with pytest.raises(ProviderResponseError):
        await model.call_tool("system", "user", _TOOL_SCHEMA)


async def test_call_tool_raises_provider_response_error_on_malformed_json() -> None:
    model = LLMModel(api_key="sk-or-test-dummy")
    tool_call = SimpleNamespace(
        function=SimpleNamespace(name="extract_cv_facts", arguments="{not valid json")
    )
    message = SimpleNamespace(tool_calls=[tool_call])
    model.client = MagicMock(name="openai_client")
    model.client.chat.completions.create = AsyncMock()
    model.client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=message)]
    )

    with pytest.raises(ProviderResponseError):
        await model.call_tool("system", "user", _TOOL_SCHEMA)
