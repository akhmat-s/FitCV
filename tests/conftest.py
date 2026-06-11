"""Shared pytest fixtures for the extract-pass test suite.

The OpenRouter client is mocked in every test — no live API calls are made.
`mock_llm` exposes a stand-in `LLMModel` whose `call_tool` returns
caller-supplied tool-call JSON, so unit/integration tests drive deterministic facts
and JD analysis without touching the network.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture(autouse=True)
def _no_live_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure no real OpenRouter key leaks into tests; force a dummy server-side key."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-dummy")


@pytest.fixture
def mock_llm() -> MagicMock:
    """A mocked OpenRouter function-calling client.

    `call_tool` is an `AsyncMock` (the real `LLMModel.call_tool` is now a coroutine the
    pipeline awaits); tests set `mock_llm.call_tool.return_value` (or `.side_effect`) to the
    tool-call dict the model would have returned, exactly as before — `AsyncMock` returns
    that dict when awaited, runs a sync `side_effect` router (returning its dict), and raises
    an exception `side_effect`. No HTTP is performed. The mock records call counts so tests
    can assert a single shared extract (e.g. ``analyze_jd`` called exactly once).
    """
    client = MagicMock(name="LLMModel")
    client.call_tool = AsyncMock(name="call_tool")
    return client
