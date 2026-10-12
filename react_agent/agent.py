"""The ReAct loop with production guardrails.

Loop shape:  think -> act -> observe -> reflect
Guardrails:
- max_steps hard limit (agent stops, returns best-effort answer)
- repeated identical actions are detected -> reflection injected
- tool errors become Observations (the model can recover, not crash)
- parse errors become Observations (formatting self-correction)
- self-critique: when an action fails, the agent critiques its own plan
  once and retries with the critique appended to the context
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

LLM = Callable[[list[dict[str, str]]], str]
ToolFn = Callable[[dict[str, Any]], Any]


class StepType(Enum):
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    REFLECTION = "reflection"
    FINAL = "final"


@dataclass
class Step:
    type: StepType
    content: str
    action: str | None = None
    args: dict[str, Any] | None = None
    result: Any = None
    ts: float = field(default_factory=time.time)


@dataclass
class RunResult:
    answer: str | None
    status: str  # "completed" | "max_steps" | "no_tools"
    steps: list[Step]
    reflections: int = 0

    @property
    def ok(self) -> bool:
        return self.status == "completed"


SYSTEM_PROMPT = """You are a ReAct planning agent. Solve the task step by step.

Available tools:
{tools}

Respond with EXACTLY ONE step per turn, in one of these formats:

Thought: <your reasoning about what to do next>
Action: <tool name>
Action Input: <JSON object of arguments>

or, when you know the final answer:

Thought: <why you are done>
Final Answer: <the answer>

Rules:
- One step per reply. No extra text before or after.
- If an Observation reports an error, think about why, then choose a
  different approach.
- Never repeat an action with identical arguments; it will fail again."""


@dataclass
class ReActAgent:
    llm: LLM
    tools: dict[str, ToolFn] = field(default_factory=dict)
    max_steps: int = 8
    reflect_after_identical: int = 2  # 2nd identical action triggers critique
    max_reflections: int = 2

    trace: list[Step] = field(default_factory=list)

    def register(self, name: str, fn: ToolFn) -> None:
        self.tools[name] = fn

    # ------------------------------------------------------------------
    def run(self, task: str) -> RunResult:
        if not self.tools:
            return RunResult(None, "no_tools", self.trace)

        from .parser import ParseError, parse_step

        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT.format(tools=self._describe_tools())},
            {"role": "user", "content": f"Task: {task}"},
        ]
        action_counts: dict[tuple[str, str], int] = {}
        reflections = 0

        for _ in range(self.max_steps):
            raw = self.llm(messages)
            try:
                parsed = parse_step(raw)
            except ParseError as exc:
                obs = f"ParseError: {exc}. Reply with exactly one step: " \
                      "Thought/Action/Action Input or Thought/Final Answer."
                self._record(StepType.OBSERVATION, obs)
                messages += [{"role": "assistant", "content": raw},
                             {"role": "user", "content": obs}]
                continue

            if parsed.thought:
                self._record(StepType.THOUGHT, parsed.thought)

            if parsed.kind == "final":
                self._record(StepType.FINAL, parsed.answer or "")
                return RunResult(parsed.answer, "completed", self.trace, reflections)

            # -- act ------------------------------------------------------
            action, args = parsed.action, parsed.args or {}
            key = (action, _stable_repr(args))
            action_counts[key] = action_counts.get(key, 0) + 1

            if action not in self.tools:
                obs = f"Error: unknown tool {action!r}. Available: {', '.join(self.tools)}"
                self._record(StepType.OBSERVATION, obs, action=action)
            else:
                try:
                    result = self.tools[action](args)
                    self._record(StepType.OBSERVATION, str(result), action=action,
                                 args=args, result=result)
                    obs = f"Observation: {result}"
                except Exception as exc:  # tool failure -> recoverable observation
                    self._record(StepType.OBSERVATION, f"ToolError: {exc}",
                                 action=action, args=args)
                    obs = f"Observation: ToolError: {exc}"

                    # self-critique on failure (bounded)
                    if reflections < self.max_reflections:
                        reflections += 1
                        critique = self._self_critique(task, action, args, str(exc))
                        self._record(StepType.REFLECTION, critique)
                        obs += f"\nReflection on failure: {critique}"

            # repeated-action reflection (bounded)
            if action_counts[key] >= self.reflect_after_identical and reflections < self.max_reflections:
                reflections += 1
                critique = ("You have already taken this exact action before with the same "
                            "arguments and it did not finish the task. Critique your plan: "
                            "is there a different tool or argument that gets new information?")
                self._record(StepType.REFLECTION, critique)
                obs += f"\nRepeated-action warning: {critique}"

            messages += [{"role": "assistant", "content": raw},
                         {"role": "user", "content": obs}]

        # graceful degradation: out of steps -> best-effort summary answer
        summary = self._best_effort_answer(task, messages)
        return RunResult(summary, "max_steps", self.trace, reflections)

    # ------------------------------------------------------------------
    def _self_critique(self, task: str, action: str, args: dict, error: str) -> str:
        """Ask the LLM to critique its own failed action (one short call)."""
        prompt = (
            f"Task: {task}\nThe action `{action}` with args {args} failed: {error}\n"
            "In one sentence: what is the likely cause and what different action "
            "or arguments should be tried instead?"
        )
        try:
            return self.llm([{"role": "user", "content": prompt}])[:300]
        except Exception:
            return f"Action {action} failed with {error[:120]}; try a different approach."

    def _best_effort_answer(self, task: str, messages: list[dict]) -> str:
        """Graceful degradation: partial answer from what was learned so far."""
        digest = "\n".join(
            f"- {s.type.value}: {s.content[:150]}" for s in self.trace[-8:]
        )
        prompt = (
            f"You attempted the task below but hit the step limit.\n"
            f"Recent trace:\n{digest}\n\n"
            "Give the best-effort partial answer based ONLY on the trace, "
            "prefixed with 'Best effort (step limit reached): '. If truly "
            "nothing was learned, say so plainly."
        )
        try:
            return self.llm([{"role": "user", "content": prompt}])[:500]
        except Exception:
            return "Best effort (step limit reached): no answer produced."

    def _describe_tools(self) -> str:
        return "\n".join(f"- {name}" for name in self.tools)

    def _record(self, type_: StepType, content: str, **kw) -> None:
        self.trace.append(Step(type=type_, content=content, **kw))


def _stable_repr(args: dict[str, Any]) -> str:
    import json

    return json.dumps(args, sort_keys=True, default=str)
