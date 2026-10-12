# ReAct Planning Agent

**Project 3 / 12 — Agentic AI portfolio.**
*Shows: you can build agents that don't infinite loop.*

A ReAct agent (observe → think → act → reflect) with the guardrails that
separate a demo from a production loop: hard iteration limits, bounded
self-critique on failure, repeated-action detection, error recovery, and
graceful degradation to a best-effort answer when the step budget runs
out.

## Why this matters

Textbook ReAct loops fail in four predictable ways: the model outputs
malformed steps, calls a tool with the same args over and over, crashes
on a tool error, or burns tokens forever without converging. Each one
gets a specific defense here:

| Failure mode | Defense |
| --- | --- |
| Malformed step | `ParseError` becomes an Observation; the model corrects its own format next turn |
| Identical repeated action | Detected by (action, args) signature; reflection injected, repeat warned |
| Tool raises | Error becomes an Observation; bounded self-critique asks the LLM why it failed and what to try instead |
| Never converges | `max_steps` hard limit; agent degrades to a best-effort partial answer from the trace |

## Architecture

```
Task ──> ReAct loop (max_steps):
           LLM ──parse_step()──┬─ Thought + Action + Action Input
                                │        │
                                │     tool(args) ──ok──> Observation ─┐
                                │        │error                     │
                                │        └─> ToolError Observation  │
                                │             + self-critique      │
                                └─ Thought + Final Answer ──> done ◄┘
           step limit hit ──> best-effort answer from trace (graceful)
```

- `react_agent/parser.py` — strict Thought/Action/Final Answer parser with
  lenient fallbacks (missing Thought lines) and scalar input normalization
- `react_agent/agent.py` — the loop: trace of every step, action
  signature counting, bounded reflections (`max_reflections`), best-effort
  degradation
- `react_agent/llm.py` — OpenAI-compatible live LLM wiring

## Quickstart

```bash
uv pip install -e ".[demo]"
python demo.py            # deterministic scripted run, no network
python demo.py --live     # + live LLM run against the same task
```

The demo task: a fictional logistics dataset (3 customers) where the agent
must chain `list_customers` → `get_customer` to answer a comparison
question. The scripted run also shows a tool failure triggering
self-critique and recovery.

## Tests

```bash
pytest -q
```

13 tests cover: parser formats (action, final, missing-thought fallbacks,
scalar input, garbage rejection), two-tool chains, tool-error recovery
with reflection, parse-error recovery, max-steps graceful degradation with
repeated-action critique, unknown-tool observation, no-tools guard, and
zero-step-limit behavior.

## Design decisions

- **Errors are observations, never crashes** — a tool exception or parse
  failure is converted into text the model can read and react to; the
  loop keeps its invariants.
- **Reflections are bounded** — self-critique costs an extra LLM call per
  failure; capping it (`max_reflections`) prevents critique loops, a
  failure mode as real as action loops.
- **Action signatures for repeat detection** — `json.dumps(args,
  sort_keys=True)` gives stable keys, so `{"a":1,"b":2}` and
  `{"b":2,"a":1}` count as the same action.
- **Degradation over silence** — hitting the step limit returns the best
  partial answer with an explicit prefix, rather than an exception or a
  fabricated confident answer.
- **Zero-dependency core** — stdlib only; the LLM is any
  `messages -> text` callable.

## Portfolio context

Project 3 of a 12-project agentic-AI portfolio (structured output, RAG
grounding, ReAct planning, tool orchestration, memory, HITL approval,
cost routing, event automation, debate, self-eval, observability, OSS
contribution). See the profile README for the full map.
