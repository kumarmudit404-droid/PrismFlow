# /part20 — Angle Reasoners (PrismFlow V2)

**Depends:** Part 19 (AngleEvidence)  
**Estimate:** 6 hours  
**Difficulty:** High  
**Gate:** All 5 reasoners callable; LLM responses parsed; no prompt injection

---

## Context

PrismFlow V2 uses **one LLM per angle** to ensure independence. Each reasoner:
1. Receives `AngleEvidence` (ranked records from Part 19)
2. Calls a distinct LLM endpoint (alternating providers: Claude, OpenAI, Claude, OpenAI, Claude)
3. Generates structured claims with confidence, citations, conflicts, caveats
4. Returns `ClaimSet` with audit trail

**Providers:** Alternate to reduce correlation:
- Tech → Claude (Sonnet 4.6)
- Market → OpenAI (GPT-4o)
- Financial → Claude (Sonnet 4.6)
- Regulatory → OpenAI (GPT-4o)
- Sentiment → Claude (Sonnet 4.6)

---

## Goals

1. Define `Claim` dataclass:
   - text: str (assertion, e.g., "TensorFlow adoption increased 12% YoY")
   - confidence: float [0, 1]
   - cited_ids: List[str] (references to evidence records by id)
   - conflicts_noted: List[str] (conflicting claims in evidence)
   - caveats: List[str] (limitations, e.g., "based on single source", "preliminary data")

2. Define `ClaimSet` dataclass:
   - angle_name: str
   - query_text: str
   - claims: List[Claim]
   - provider: str ("claude" or "openai")
   - model_name: str (e.g., "claude-sonnet-4-6", "gpt-4o")
   - tokens_input: int
   - tokens_output: int
   - latency_seconds: float
   - audit_trail: List[str] (logs of reasoning steps)

3. Implement `AngleReasoner` ABC with:
   - `reason(angle_evidence: AngleEvidence) -> ClaimSet`
   - Provider-specific subclasses: `ClaudeReasoner`, `OpenAIReasoner`
   - Prompt injection defense: XML-wrapped evidence, sanitized user text
   - Structured output parsing (JSON)

4. Write per-angle system prompts (in `prompts/v2/angle_*.txt`):
   - Angle-specific framing (e.g., "you are a tech analyst")
   - Evidence presentation format (XML-wrapped)
   - Output schema (JSON with Claim fields)
   - Explicit instructions on conflicts, caveats, citations

5. Tests:
   - Each reasoner parses LLM output into ClaimSet
   - Cited IDs match evidence records
   - Confidence scores in [0, 1]
   - Prompt injection defense (adversarial strings don't break parsing)

---

## Specifications

### `Claim` and `ClaimSet` Models

```python
# prismflow/v2/reasoners/models.py

from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime

@dataclass
class Claim:
    text: str
    confidence: float             # [0, 1]
    cited_ids: List[str] = field(default_factory=list)
    conflicts_noted: List[str] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence {self.confidence} not in [0, 1]")

@dataclass
class ClaimSet:
    angle_name: str
    query_text: str
    claims: List[Claim]
    provider: str                 # "claude" or "openai"
    model_name: str               # "claude-sonnet-4-6", "gpt-4o"
    tokens_input: int
    tokens_output: int
    latency_seconds: float
    audit_trail: List[str] = field(default_factory=list)
    
    def total_claims(self) -> int:
        return len(self.claims)
```

### `AngleReasoner` ABC and Implementations

```python
# prismflow/v2/reasoners/base.py

from abc import ABC, abstractmethod
from prismflow.v2.angles.models import AngleEvidence
from prismflow.v2.reasoners.models import ClaimSet
import logging

logger = logging.getLogger(__name__)

class AngleReasoner(ABC):
    def __init__(self, angle_name: str):
        self.angle_name = angle_name
    
    @abstractmethod
    async def reason(self, evidence: AngleEvidence) -> ClaimSet:
        """Generate claims from angle evidence."""
        pass
    
    def _sanitize_input(self, text: str) -> str:
        """Remove potentially malicious markup from user/evidence text."""
        # Remove XML-like tags except our own <evidence> tags
        import re
        # Allow only alphanumeric, spaces, common punctuation
        return re.sub(r'[<>{}]', '', text)
    
    def _format_evidence_xml(self, evidence: AngleEvidence) -> str:
        """Format evidence as safe XML for LLM."""
        xml = f"""<angle_evidence angle="{self.angle_name}">
  <query>{self._sanitize_input(evidence.query_text)}</query>
  <records>"""
        for i, record in enumerate(evidence.ranked_records[:10]):  # Top 10
            xml += f"""
    <record id="{record.id}">
      <title>{self._sanitize_input(record.title)}</title>
      <url>{self._sanitize_input(record.url)}</url>
      <snippet>{self._sanitize_input(record.snippet_tokens[:200])}</snippet>
      <author>{self._sanitize_input(record.author or 'unknown')}</author>
      <score>{record.relevance_score:.2f}</score>
    </record>"""
        xml += """
  </records>
</angle_evidence>"""
        return xml
```

### Claude Reasoner

```python
# prismflow/v2/reasoners/claude_reasoner.py

import asyncio
import json
import time
from typing import List, Optional
from anthropic import Anthropic
from prismflow.v2.reasoners.base import AngleReasoner
from prismflow.v2.reasoners.models import Claim, ClaimSet
from prismflow.v2.angles.models import AngleEvidence

class ClaudeReasoner(AngleReasoner):
    def __init__(self, angle_name: str, model: str = "claude-sonnet-4-6"):
        super().__init__(angle_name)
        self.model = model
        self.client = Anthropic()
    
    async def reason(self, evidence: AngleEvidence) -> ClaimSet:
        """Generate claims using Claude."""
        start_time = time.time()
        audit_trail = []
        
        # Load angle-specific system prompt
        system_prompt = self._load_system_prompt()
        audit_trail.append(f"Loaded system prompt for angle '{self.angle_name}'")
        
        # Format evidence as XML
        evidence_xml = self._format_evidence_xml(evidence)
        audit_trail.append(f"Formatted {len(evidence.ranked_records)} records as XML")
        
        # Construct user message
        user_message = f"""Analyze the following {self.angle_name} evidence and generate claims.

{evidence_xml}

Generate your response as JSON matching this schema:
{{
  "claims": [
    {{
      "text": "...",
      "confidence": 0.85,
      "cited_ids": ["record_id_1", "record_id_2"],
      "conflicts_noted": ["..."],
      "caveats": ["..."]
    }}
  ]
}}
"""
        
        # Call Claude
        audit_trail.append(f"Calling Claude {self.model}")
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_message}
            ]
        )
        
        # Parse response
        response_text = response.content[0].text
        audit_trail.append(f"Received {response.usage.output_tokens} output tokens")
        
        # Extract JSON
        try:
            # Find JSON block in response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in response")
            
            json_str = json_match.group()
            claims_data = json.loads(json_str)
            audit_trail.append("Parsed JSON response")
        except json.JSONDecodeError as e:
            audit_trail.append(f"JSON parse error: {e}")
            claims_data = {"claims": []}
        
        # Build Claim objects
        claims = []
        for claim_data in claims_data.get("claims", []):
            claim = Claim(
                text=claim_data.get("text", ""),
                confidence=float(claim_data.get("confidence", 0.5)),
                cited_ids=claim_data.get("cited_ids", []),
                conflicts_noted=claim_data.get("conflicts_noted", []),
                caveats=claim_data.get("caveats", [])
            )
            claims.append(claim)
        
        latency = time.time() - start_time
        
        return ClaimSet(
            angle_name=self.angle_name,
            query_text=evidence.query_text,
            claims=claims,
            provider="claude",
            model_name=self.model,
            tokens_input=response.usage.input_tokens,
            tokens_output=response.usage.output_tokens,
            latency_seconds=latency,
            audit_trail=audit_trail
        )
    
    def _load_system_prompt(self) -> str:
        """Load angle-specific system prompt from file."""
        try:
            with open(f"prompts/v2/angle_{self.angle_name}.txt", "r") as f:
                return f.read()
        except FileNotFoundError:
            return f"You are a {self.angle_name} analysis expert. Generate well-reasoned claims."
```

### OpenAI Reasoner (similar structure)

```python
# prismflow/v2/reasoners/openai_reasoner.py

import asyncio
import json
import time
from openai import AsyncOpenAI
from prismflow.v2.reasoners.base import AngleReasoner
from prismflow.v2.reasoners.models import Claim, ClaimSet

class OpenAIReasoner(AngleReasoner):
    def __init__(self, angle_name: str, model: str = "gpt-4o"):
        super().__init__(angle_name)
        self.model = model
        self.client = AsyncOpenAI()
    
    async def reason(self, evidence: AngleEvidence) -> ClaimSet:
        # Similar implementation, using OpenAI client
        # (see Claude implementation above for structure)
        pass
```

### Angle-Specific System Prompts

```text
# prompts/v2/angle_tech.txt

You are a technology analysis expert specializing in software, infrastructure, and AI/ML trends.

Your task: Analyze the provided evidence (research papers, GitHub repositories, technical news) and generate 3-5 key claims about technology trends.

Requirements:
1. Each claim must cite at least one evidence record (by id).
2. Confidence should reflect certainty: 0.9+ for consensus findings, 0.5-0.7 for emerging/contested claims.
3. Note conflicting evidence if sources disagree.
4. Include caveats (e.g., "limited to X domain", "based on preliminary data").
5. Avoid speculation beyond the evidence.

Output JSON with schema: {"claims": [...]}
```

Similar files for market, financial, regulatory, sentiment angles.

---

## Deliverables

### File structure
```
prismflow/v2/
├── reasoners/
│   ├── __init__.py
│   ├── base.py                 # AngleReasoner ABC
│   ├── models.py               # Claim, ClaimSet
│   ├── claude_reasoner.py      # ClaudeReasoner
│   ├── openai_reasoner.py      # OpenAIReasoner
│   └── factory.py              # get_reasoner(angle_name) -> AngleReasoner

prompts/v2/
├── angle_tech.txt
├── angle_market.txt
├── angle_financial.txt
├── angle_regulatory.txt
└── angle_sentiment.txt

tests/v2/
├── test_reasoners.py           # Per-reasoner tests
```

### Tests to write

1. **test_claude_reasoner_parses_claims()** — Mock Anthropic response, verify ClaimSet
2. **test_openai_reasoner_parses_claims()** — Mock OpenAI response
3. **test_cited_ids_exist()** — Verify all cited_ids are in evidence.ranked_records
4. **test_confidence_in_range()** — Verify all confidence scores in [0, 1]
5. **test_prompt_injection_defense()** — Evidence with `<script>alert('xss')</script>` doesn't break parsing
6. **test_reasoner_factory()** — get_reasoner("tech") returns ClaudeReasoner, get_reasoner("market") returns OpenAIReasoner

---

## Standing Rules

1. **No V1 modifications.**
2. **Async only.** All reasoners must be async for concurrent Part 23 fusion.
3. **Prompt injection.** Sanitize evidence input; wrap in XML to mark boundaries.
4. **JSON parsing.** Use regex to extract JSON if LLM embeds it in prose.
5. **Error resilience.** If parse fails, return ClaimSet with 0 claims; log error.

---

## Success Criteria

- [ ] ClaudeReasoner and OpenAIReasoner implemented
- [ ] All 5 angles have system prompts
- [ ] JSON parsing handles edge cases (malformed JSON, missing fields)
- [ ] Prompt injection test passes
- [ ] All tests pass: `pytest tests/v2/test_reasoners.py -v`
- [ ] Ready for Part 21

---

## Checklist for Closing

- [ ] Reasoners callable: `await claude_reasoner.reason(evidence)`
- [ ] Manual test: call reasoner with sample evidence, log ClaimSet
- [ ] Tests pass: `pytest tests/v2/test_reasoners.py -v`
