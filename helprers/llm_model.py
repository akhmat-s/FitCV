"""OpenRouter (OpenAI-compatible) function-calling client.

Replaces the former local llama.cpp model. Exposes a single function-calling call:
given a system prompt, a user prompt, and a tool schema, force the model to return a
tool call and parse its arguments into a dict (consumed by `extract.extract_cv_facts`
and `extract.analyze_jd`).

The API key is read server-side from OPENROUTER_API_KEY and is never logged or
returned. This client lets openai provider errors (timeout/network/auth) propagate;
``extract.build_extract`` catches and translates them to an extract-stage
ErrorResponse with the key redacted.

The client is ``AsyncOpenAI`` and ``call_tool`` is a coroutine awaited up the
pipeline. ``AsyncOpenAI`` raises the SAME exception hierarchy
(``APITimeoutError``/``APIConnectionError``/``RateLimitError``/``AuthenticationError``/
``APIStatusError``/``OpenAIError``) and honors the same per-request ``timeout=`` as the
former sync client, so the ``extract._provider_error`` mapping and the ``{error, stage}``
envelope are unchanged.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable

from openai import AsyncOpenAI

from schemas import DEFAULT_MODEL_NAME, DEFAULT_OPENROUTER_BASE_URL, EXTRACT_TIMEOUT_S


async def bounded_gather(coros: list[Awaitable], semaphore: asyncio.Semaphore) -> list:
    """Order-preserving ``asyncio.gather`` that runs at most ``semaphore`` slots concurrently.

    Each coroutine acquires the shared ``semaphore`` before it runs, so at most N LLM-issuing
    coroutines are ever in flight at once (rate-limit safety; the ``MAX_CONCURRENT_LLM_CALLS`` cap).
    Results are returned in the SAME order as ``coros`` — never completion order — so callers map
    them back to a fixed key/index and determinism holds regardless of which finishes first.
    """

    async def _run(coro: Awaitable):
        async with semaphore:
            return await coro

    return await asyncio.gather(*(_run(coro) for coro in coros))


class ProviderResponseError(Exception):
    """Raised when the provider's response can't be parsed into tool arguments.

    The model is forced via ``tool_choice`` to emit exactly one tool call, but a model
    or proxy can still return an empty/malformed response or non-JSON arguments. Raising
    a typed error lets ``extract.build_extract`` translate it to a ``stage=extract``
    ErrorResponse instead of crashing with a raw provider traceback.
    """


class LLMModel:
    """Thin OpenRouter function-calling client over the OpenAI-compatible SDK."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = EXTRACT_TIMEOUT_S,
    ) -> None:
        # API key server-side only. Never log this value.
        # Model / base URL are read from the environment HERE (at construction),
        # not frozen at schemas import, so values placed in .env (loaded at app entry)
        # are honored. Falls back to the schemas defaults when unset.
        self._api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model or os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
        self.timeout = timeout
        self.client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=base_url or os.getenv("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL),
        )

    async def call_tool(
        self,
        system_prompt: str,
        user_prompt: str,
        tool_schema: dict,
        timeout: float | None = None,
    ) -> dict:
        """Force a single function call and return its parsed arguments as a dict.

        Args:
            system_prompt: Instruction block for the model.
            user_prompt: The CV or JD text to analyze.
            tool_schema: The function-calling tool schema.
            timeout: Per-call ceiling; defaults to ``EXTRACT_TIMEOUT_S``.

        Returns:
            The parsed tool-call arguments as a ``dict``.
        """
        # Force exactly this tool via tool_choice so the model is guaranteed to emit
        # structured JSON, then json.loads its arguments.
        tool_name = tool_schema["name"]
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tools=[{"type": "function", "function": tool_schema}],
            tool_choice={"type": "function", "function": {"name": tool_name}},
            timeout=timeout if timeout is not None else self.timeout,
        )
        # tool_choice does not guarantee a well-formed response from every model/proxy.
        # Guard the response shape and the JSON parse so a malformed reply surfaces as a
        # typed ProviderResponseError (translated to a stage=extract ErrorResponse
        # upstream) rather than a raw IndexError/TypeError/JSONDecodeError.
        choices = response.choices
        if not choices:
            raise ProviderResponseError("The provider returned no choices.")
        tool_calls = choices[0].message.tool_calls
        if not tool_calls:
            raise ProviderResponseError("The provider returned no tool call.")
        try:
            return json.loads(tool_calls[0].function.arguments)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ProviderResponseError(
                "The provider returned malformed tool arguments."
            ) from exc
