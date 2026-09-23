"""Claude-backed reasoner.

Part 20. Only the request differs from the base class; parsing, validation and
the injection defences are inherited.

THE CLIENT MUST BE ASYNC
------------------------
The brief declares ``async def reason`` and then calls the SYNCHRONOUS
``Anthropic().messages.create(...)`` inside it. That blocks the event loop for
the whole request, so Part 23's five concurrent angles would run one after
another -- and nothing else on the loop (Part 18's cache, Part 19's retrieval)
could progress either. ``AsyncAnthropic`` is used instead, which is what
standing rule 2 requires in practice rather than only in signature.

THE MODEL ID
------------
The brief names ``claude-sonnet-4-6``, which is a real and still-served model.
The default here is ``claude-sonnet-5``: it is the current generation of the
tier the brief asked for and it is cheaper ($2/$10 per MTok against $3/$15), so
it is better on both axes the choice trades off. It is one constant, and any
caller can pass ``model=`` to override it.

READING THE RESPONSE
--------------------
The brief reads ``response.content[0].text``. Content is a list of typed blocks,
and with thinking enabled the first block is a ``thinking`` block, which has no
``.text`` -- an AttributeError on the first call. Every text block is collected
here instead, by checking ``block.type`` first.

Thinking is on by default (adaptive): the task is to weigh partly conflicting
evidence and assign calibrated confidence, which is the kind of judgement it
helps. It costs output tokens, which is why ``max_output_tokens`` defaults well
above the brief's 1024 -- thinking tokens count against the same ceiling, and a
reply cut mid-JSON parses to nothing.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .base import AngleReasoner, ProviderResponse, ReasonerError

logger = logging.getLogger("prismflow.v2.reasoners")

#: Current generation of the tier the brief specifies. See the module docstring.
DEFAULT_CLAUDE_MODEL = "claude-sonnet-5"


class ClaudeReasoner(AngleReasoner):
    """An angle reasoner backed by the Anthropic Messages API."""

    PROVIDER = "claude"

    def __init__(
        self,
        angle_name: str,
        *,
        model: Optional[str] = None,
        client: Any = None,
        thinking: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(angle_name, model=model, **kwargs)
        self._client = client
        self.thinking = thinking

    @classmethod
    def default_model(cls) -> str:
        return DEFAULT_CLAUDE_MODEL

    @property
    def client(self) -> Any:
        """The Anthropic client, created on first use.

        Lazy so a reasoner can be constructed, inspected and unit-tested in an
        environment with no credentials -- which is this repository's state, and
        the state of any fresh checkout.
        """
        if self._client is None:
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic()
        return self._client

    async def _complete(self, system: str, user: str) -> ProviderResponse:
        import anthropic

        request: dict = {
            "model": self.model,
            "max_tokens": self.max_output_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if self.thinking:
            request["thinking"] = {"type": "adaptive"}

        try:
            response = await self.client.messages.create(**request)
        except anthropic.APIStatusError as exc:
            raise ReasonerError(
                f"{type(exc).__name__} {getattr(exc, 'status_code', '?')}: "
                f"{getattr(exc, 'message', exc)}"
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise ReasonerError(f"connection error: {exc}") from exc
        except anthropic.AnthropicError as exc:
            raise ReasonerError(f"{type(exc).__name__}: {exc}") from exc

        text = _text_of(response)
        usage = getattr(response, "usage", None)
        stop_reason = getattr(response, "stop_reason", None)
        return ProviderResponse(
            text=text,
            tokens_input=int(getattr(usage, "input_tokens", 0) or 0),
            tokens_output=int(getattr(usage, "output_tokens", 0) or 0),
            model_name=getattr(response, "model", self.model) or self.model,
            stop_reason=stop_reason,
            truncated=stop_reason == "max_tokens",
        )


def _text_of(response: Any) -> str:
    """Concatenate every text block, skipping thinking and tool blocks."""
    parts = []
    for block in getattr(response, "content", None) or []:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", "") or "")
    return "\n".join(part for part in parts if part)


__all__ = ["ClaudeReasoner", "DEFAULT_CLAUDE_MODEL"]
