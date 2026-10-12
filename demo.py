"""Demo: a ReAct agent working through a small multi-step task with a
scripted (deterministic) LLM, then optionally a live LLM.

Task universe: a tiny fictional logistics dataset -- customers, orders,
distances -- and tools to query it. The agent must chain multiple tool
calls to answer "which customer is farthest from the warehouse, and what
is their open order total?"

Usage:
  python demo.py            # scripted, no network
  python demo.py --live     # + live LLM run
"""

from __future__ import annotations

import sys

from react_agent.agent import ReActAgent, StepType

# ---- fictional data ------------------------------------------------
CUSTOMERS = {
    "C-1": {"name": "North Traders", "distance_km": 120, "open_total": 4300.0},
    "C-2": {"name": "South Supplies", "distance_km": 310, "open_total": 8100.0},
    "C-3": {"name": "East Export Co", "distance_km": 240, "open_total": 15600.0},
}


def tool_list_customers(args):
    return sorted(CUSTOMERS)


def tool_get_customer(args):
    cid = args.get("id") or args.get("input")
    if cid not in CUSTOMERS:
        raise ValueError(f"unknown customer {cid!r}; use list_customers first")
    return CUSTOMERS[cid]


def tool_add(args):
    return args["a"] + args["b"]


TOOLS = {
    "list_customers": tool_list_customers,
    "get_customer": tool_get_customer,
    "add": tool_add,
}


def scripted(replies):
    def llm(messages):
        return replies.pop(0)
    return llm


def print_trace(agent):
    icons = {StepType.THOUGHT: "THINK", StepType.ACTION: "ACT  ",
             StepType.OBSERVATION: "OBS  ", StepType.REFLECTION: "REFLT",
             StepType.FINAL: "FINAL"}
    for s in agent.trace:
        print(f"  [{icons[s.type]}] {s.content[:120]}")


def main():
    print("=" * 62)
    print("Scripted run: 'Which customer is farthest from the warehouse,")
    print("and what is their open order total?'")
    print("=" * 62)
    llm = scripted([
        "Thought: I need to see all customers and their distances first.\n"
        "Action: list_customers\nAction Input: {}",
        "Thought: C-2 (South Supplies) at 310 km is farthest. Now get their open total.\n"
        'Action: get_customer\nAction Input: {"id": "C-2"}',
        "Thought: I have the distance (310 km) and the open total (8100).\n"
        "Final Answer: South Supplies (C-2) is farthest at 310 km, with an open order total of 8100.00",
    ])
    agent = ReActAgent(llm, tools=dict(TOOLS), max_steps=8)
    r = agent.run("Which customer is farthest from the warehouse, and what is their open order total?")
    print_trace(agent)
    print(f"\nstatus={r.status}  answer={r.answer}")

    print()
    print("=" * 62)
    print("Scripted run with failure + recovery: tool error -> self-critique")
    print("=" * 62)
    llm2 = scripted([
        "Thought: Look up the mystery customer directly.\n"
        'Action: get_customer\nAction Input: {"id": "C-99"}',
        "Thought: The id was wrong; list the customers first, then pick the right id.\n"
        "Action: list_customers\nAction Input: {}",
        "Thought: Now fetch C-1 with the correct id.\n"
        'Action: get_customer\nAction Input: {"id": "C-1"}',
        "Thought: Done.\nFinal Answer: North Traders is 120 km away with 4300.00 open.",
    ])
    agent2 = ReActAgent(llm2, tools=dict(TOOLS), max_steps=8)
    r2 = agent2.run("What is North Traders' distance and open total?")
    print_trace(agent2)
    print(f"\nstatus={r2.status}  answer={r2.answer}")

    if "--live" in sys.argv:
        print()
        print("=" * 62)
        print("Live LLM run (same task)")
        print("=" * 62)
        from react_agent.llm import make_llm
        try:
            live = make_llm()
        except RuntimeError as e:
            print(f"skipping live demo: {e}")
            return
        agent3 = ReActAgent(live, tools=dict(TOOLS), max_steps=10)
        r3 = agent3.run("Which customer is farthest from the warehouse, and what is their open order total?")
        print_trace(agent3)
        print(f"\nstatus={r3.status}  answer={r3.answer}")


if __name__ == "__main__":
    main()
