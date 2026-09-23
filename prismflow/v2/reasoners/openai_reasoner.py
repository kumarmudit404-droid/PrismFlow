"""OpenAI-backed reasoner.

Part 20. The brief leaves this one as ``pass``; it is implemented here to the
same contract as the Claude reasoner, which is possible in a few lines precisely
because everything except the request lives in the base class.

TOKEN FIELDS DIFFER BETWEEN THE TWO SDKS
----------------------------------------
Anthropic reports ``usage.input_tokens`` / ``usage.output_tokens``; OpenAI
reports ``usage.prompt_tokens`` / ``usage.completion_tokens``. Copying the
Claude implementation "in the same shape", as the brief suggests, silently
yields zeros for this provider -- and a ClaimSet that reports zero tokens looks
like a free call rather than a mismeasured one.

The client is constructed lazily for a harder reason than Claude's: ``OpenAI()``
and ``AsyncOpenAI()`` RAISE at construction when no credential is present
(``OpenAIError: Missing credentials``), where ``AsyncAnthropic()`` does not. A
module that built its client eagerly could not even be imported on a machine
without an OpenAI key, which would take the reasoner factory, its tests and
every angle down with it.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .base import AngleReasoner, ProviderResponse, ReasonerError

logger = logging.getLogger("prismflow.v2.reasoners")

#: The model the brief names for the OpenAI angles.
DEFAULT_OPENAI_MODEL = "gpt-4o"


class OpenAIReasoner(AngleReasoner):
    """An angle reasoner backed by the OpenAI chat completions API."""

    PROVIDER = "openai"

    def __init__(
        self,
        angle_name: str,
        *,
        model: Optional[str] = None,
        client: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(angle_name, model=model, **kwargs)
        self._client = client

    @classmethod
    def default_model(cls) -> str:
        return DEFAULT_OPENAI_MODEL

    @property
    def client(self) -> Any:
        """The OpenAI client, created on first use. See the module docstring."""
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI()
        return self._client

    async def _complete(self, system: str, user: str) -> ProviderResponse:
        import openai

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_output_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
            )
        except openai.APIStatusError as exc:
            raise ReasonerError(
                f"{type(exc).__name__} {getattr(exc, 'status_code', '?')}: {exc}"
            ) from exc
        except openai.APIConnectionError as exc:
            raise ReasonerError(f"connection error: {exc}") from exc
        except openai.OpenAIError as exc:
            raise ReasonerError(f"{type(exc).__name__}: {exc}") from exc

        choice = (getattr(response, "choices", None) or [None])[0]
        message = getattr(choice, "message", None)
        text = getattr(message, "content", "") or ""
        finish_reason = getattr(choice, "finish_reason", None)
        usage = getattr(response, "usage", None)
        return ProviderResponse(
            text=text,
            tokens_input=int(getattr(usage, "prompt_tokens", 0) or 0),
            tokens_output=int(getattr(usage, "completion_tokens", 0) or 0),
            model_name=getattr(response, "model", self.model) or self.model,
            stop_reason=finish_reason,
            truncated=finish_reason == "length",
        )


__all__ = ["OpenAIReasoner", "DEFAULT_OPENAI_MODEL"]
