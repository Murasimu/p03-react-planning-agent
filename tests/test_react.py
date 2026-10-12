"""Tests for the ReAct planning agent. Deterministic, offline."""

import pytest

from react_agent.agent import ReActAgent, RunResult, StepType
from react_agent.parser import ParseError, parse_step


# ---------------------------------------------------------------- parser
class TestParser:
    def test_action_step(self):
        p = parse_step(
            "Thought: I should look up the customer first.\n"
            'Action: get_customer\nAction Input: {"id": "C-42"}'
        )
        assert p.kind == "action"
        assert p.action == "get_customer"
        assert p.args == {"id": "C-42"}

    def test_final_step(self):
        p = parse_step(
            "Thought: Done, I have the total.\nFinal Answer: The total is 118.50"
        )
        assert p.kind == "final"
        assert p.answer == "The total is 118.50"

    def test_missing_thought_still_parses_action(self):
        p = parse_step('Action: lookup\nAction Input: {"q": "x"}')
        assert p.kind == "action" and p.action == "lookup"

    def test_missing_thought_still_parses_final(self):
        p = parse_step("Final Answer: 42")
        assert p.kind == "final" and p.answer == "42"

    def test_scalar_input_normalized(self):
        p = parse_step("Action: search\nAction Input: hello world")
        assert p.args == {"input": "hello world"}

    def test_garbage_raises(self):
        with pytest.raises(ParseError):
            parse_step("I think therefore I am, no structure at all")
        with pytest.raises(ParseError):
            parse_step("")


# ---------------------------------------------------------------- agent
class ScriptedLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, messages):
        self.calls.append([dict(m) for m in messages])
        return self.replies.pop(0)


def make_tools():
    return {
        "add": lambda a: a["a"] + a["b"],
        "echo": lambda a: f"echo:{a.get('input', '')}",
        "boom": lambda a: (_ for _ in ()).throw(RuntimeError("database unavailable")),
    }


def agent_with(replies, **kw):
    llm = ScriptedLLM(replies)
    agent = ReActAgent(llm, tools=make_tools(), **kw)
    return agent, llm


class TestReActAgent:
    def test_simple_two_tool_chain(self):
        agent, _ = agent_with([
            'Thought: add the numbers.\nAction: add\nAction Input: {"a": 20, "b": 22}',
            "Thought: got it.\nFinal Answer: 42",
        ])
        r = agent.run("What is 20+22?")
        assert r.ok and r.answer == "42"
        types = [s.type for s in agent.trace]
        assert StepType.THOUGHT in types and StepType.ACTION not in types
        # observation recorded for the tool call
        assert any(s.type == StepType.OBSERVATION and s.action == "add" for s in agent.trace)

    def test_tool_error_becomes_observation_and_recovers(self):
        agent, _ = agent_with([
            'Thought: try the db.\nAction: boom\nAction Input: {}',
            'Thought: db is down, use echo instead.\nAction: echo\nAction Input: {"input": "fallback"}',
            "Thought: done.\nFinal Answer: echo:fallback",
        ])
        r = agent.run("Do the thing")
        assert r.ok
        assert any(s.type == StepType.REFLECTION for s in agent.trace)  # self-critique fired

    def test_parse_error_recovers(self):
        agent, _ = agent_with([
            "Hmm I will just wing this reply without structure",  # parse error
            'Thought: fixed format.\nAction: add\nAction Input: {"a": 1, "b": 1}',
            "Thought: done.\nFinal Answer: 2",
        ])
        r = agent.run("add")
        assert r.ok and r.answer == "2"

    def test_max_steps_graceful_degradation(self):
        # Model never gives a final answer -> step limit -> best-effort answer
        loop = ['Thought: keep going.\nAction: echo\nAction Input: {"input": "x"}'] * 6
        agent, _ = agent_with(loop + ["Best effort (step limit reached): partial info"], max_steps=4)
        r = agent.run("never finishes")
        assert r.status == "max_steps"
        assert r.answer is not None  # graceful, not a crash
        assert any(s.type == StepType.REFLECTION for s in agent.trace)  # repeated-action critique

    def test_unknown_tool_observation(self):
        agent, _ = agent_with([
            'Thought: try something.\nAction: teleport\nAction Input: {}',
            "Thought: ok.\nFinal Answer: gave up teleporting",
        ])
        r = agent.run("x")
        assert r.ok
        assert any("unknown tool" in s.content for s in agent.trace if s.type == StepType.OBSERVATION)

    def test_no_tools_registered(self):
        agent = ReActAgent(ScriptedLLM([]), tools={})
        r = agent.run("anything")
        assert r.status == "no_tools"

    def test_step_limit_zero(self):
        agent, _ = agent_with(["never called"], max_steps=0)
        r = agent.run("x")
        assert r.status == "max_steps"
