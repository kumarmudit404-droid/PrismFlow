"""Angle reasoners: one LLM per angle, turning evidence into claims.

Part 20. Takes ``AngleEvidence`` from Part 19 and returns a ``ClaimSet``:
grounded assertions with confidence, citations, noted conflicts and caveats,
plus the audit trail that says how the answer was obtained.

Two things worth knowing before using this module:

- Providers alternate across angles (three Claude, two OpenAI), which reduces
  correlation between angles without removing it. ``provider_clusters`` reports
  the grouping, because Part 21 measures dependence between angles and should
  read its numbers against that structural prior.
- Claims are only as trustworthy as their citations, so every cited id is
  checked against the records the model was actually shown, and a claim left
  citing nothing is discarded. That check, not the input sanitiser, is the
  layer doing the work against prompt injection.

Downstream parts import from here, not from the submodules.
"""

from .base import (
    FALLBACK_SYSTEM_PROMPT,
    MAX_OUTPUT_TOKENS,
    MAX_RECORDS,
    MAX_SNIPPET_CHARS,
    PROMPT_DIR,
    AngleReasoner,
    ProviderResponse,
    ReasonerError,
)
from .claude_reasoner import DEFAULT_CLAUDE_MODEL, ClaudeReasoner
from .factory import (
    ANGLE_PROVIDERS,
    PROVIDER_CLASSES,
    build_reasoners,
    get_reasoner,
    provider_clusters,
)
from .models import Claim, ClaimSet
from .openai_reasoner import DEFAULT_OPENAI_MODEL, OpenAIReasoner
from .parsing import claims_payload, extract_json_object

__all__ = [
    # interface
    "AngleReasoner",
    "ProviderResponse",
    "ReasonerError",
    # providers
    "ClaudeReasoner",
    "OpenAIReasoner",
    "DEFAULT_CLAUDE_MODEL",
    "DEFAULT_OPENAI_MODEL",
    # factory
    "get_reasoner",
    "build_reasoners",
    "provider_clusters",
    "ANGLE_PROVIDERS",
    "PROVIDER_CLASSES",
    # data
    "Claim",
    "ClaimSet",
    # parsing
    "extract_json_object",
    "claims_payload",
    # config
    "PROMPT_DIR",
    "MAX_RECORDS",
    "MAX_SNIPPET_CHARS",
    "MAX_OUTPUT_TOKENS",
    "FALLBACK_SYSTEM_PROMPT",
]
