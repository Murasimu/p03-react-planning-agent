"""Strict parser for the ReAct interchange format.

The LLM must emit steps in one of these shapes:

    Thought: <reasoning>
    Action: tool_name
    Action Input: {"json": "args"}

    Thought: <reasoning>
    Final Answer: <answer>

Anything else is a ParseError -- and unlike naive ReAct loops, a parse
error is recoverable: the agent feeds the error back as an Observation so
the model can correct its formatting on the next step.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

ACTION_RE = re.compile(
    r"Thought:\s*(?P<thought>.+?)\s*\n+Action:\s*(?P<action>[A-Za-z0-9_.-]+)\s*"
    r"\n+Action Input:\s*(?P<input>.+)",
    re.DOTALL,
)
FINAL_RE = re.compile(
    r"Thought:\s*(?P<thought>.+?)\s*\n+Final Answer:\s*(?P<answer>.+)",
    re.DOTALL,
)
# Lenient fallback: model omitted the Thought line (common failure mode).
ACTION_NO_THOUGHT_RE = re.compile(
    r"Action:\s*(?P<action>[A-Za-z0-9_.-]+)\s*\n+Action Input:\s*(?P<input>.+)",
    re.DOTALL,
)
FINAL_NO_THOUGHT_RE = re.compile(r"Final Answer:\s*(?P<answer>.+)", re.DOTALL)


class ParseError(ValueError):
    """The model's step could not be parsed into Thought/Action/Answer."""


@dataclass
class ParsedStep:
    kind: str  # "action" | "final"
    thought: str | None
    action: str | None = None
    args: dict[str, Any] | None = None
    answer: str | None = None


def parse_step(text: str) -> ParsedStep:
    """Parse one ReAct step, tolerating missing Thought lines."""
    text = text.strip()
    if not text:
        raise ParseError("empty step")

    m = ACTION_RE.search(text)
    if m:
        return ParsedStep(
            kind="action",
            thought=m.group("thought").strip(),
            action=m.group("action").strip(),
            args=_parse_input(m.group("input")),
        )

    m = FINAL_RE.search(text)
    if m:
        return ParsedStep(
            kind="final",
            thought=m.group("thought").strip(),
            answer=m.group("answer").strip(),
        )

    # Lenient fallbacks
    m = ACTION_NO_THOUGHT_RE.search(text)
    if m:
        return ParsedStep(
            kind="action",
            thought=None,
            action=m.group("action").strip(),
            args=_parse_input(m.group("input")),
        )

    m = FINAL_NO_THOUGHT_RE.search(text)
    if m:
        return ParsedStep(kind="final", thought=None, answer=m.group("answer").strip())

    raise ParseError(f"unparseable step: {text[:200]!r}")


def _parse_input(raw: str) -> dict[str, Any]:
    """Action Input may be JSON or bare scalar; normalize to a dict."""
    raw = raw.strip()
    if raw.startswith("{"):
        try:
            val = json.loads(raw)
            if isinstance(val, dict):
                return val
            if isinstance(val, list):
                return {"items": val}
            return {"value": val}
        except json.JSONDecodeError:
            pass
    return {"input": raw}
