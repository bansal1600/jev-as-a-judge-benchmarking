"""Customer-support triage agent: local gemma4 + LangGraph ReAct loop.

LangSmith tracing is automatic when LANGSMITH_TRACING=true is set in .env --
no code changes needed, LangChain instruments itself.
"""
import os
import sys

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

sys.path.insert(0, os.path.dirname(__file__))
from tools import ALL_TOOLS, CATEGORIES, TEAMS  # noqa: E402

load_dotenv()

AGENT_MODEL = os.getenv("AGENT_MODEL", "gemma4-judge:latest")

SYSTEM_PROMPT = f"""You are a customer-support triage agent.

For every ticket you must:
1. If the customer mentions an order number, call lookup_order first.
2. Call categorize exactly once with one of: {', '.join(CATEGORIES)}
3. Call escalate exactly once with one of: {', '.join(TEAMS)}
4. Then write a short, empathetic reply to the customer (2-3 sentences).

Escalation guide:
- damaged_goods or return_request -> refund_team
- late_delivery -> logistics_team
- billing_issue -> billing_team
- account_access -> tier2_support
- product_question -> none

Always complete all steps before replying."""


def build_agent(model: str = AGENT_MODEL):
    """Create the ReAct agent bound to the local Ollama model."""
    llm = ChatOllama(model=model, temperature=0)
    return create_react_agent(llm, ALL_TOOLS, prompt=SYSTEM_PROMPT)


def _extract(messages):
    """Pull the structured decisions back out of the tool-call trace.

    Evaluators score these, not the prose -- that keeps the deterministic
    checks honest regardless of how the model words its reply.
    """
    category, team, looked_up = None, None, False
    for m in messages:
        for tc in getattr(m, "tool_calls", None) or []:
            name, args = tc["name"], tc.get("args", {})
            if name == "lookup_order":
                looked_up = True
            elif name == "categorize":
                category = str(args.get("category", "")).strip().lower()
            elif name == "escalate":
                team = str(args.get("team", "")).strip().lower()
    return category, team, looked_up


def _final_reply(messages) -> str:
    """Return the customer-facing prose.

    This model tends to emit all three tool calls in parallel *and* write its
    reply in that same message, leaving the post-tool message empty. So take
    the last AI message that actually carries text rather than just the last.
    """
    for m in reversed(messages):
        if type(m).__name__ == "AIMessage":
            text = str(m.content or "").strip()
            if text:
                return text
    return ""


def run_triage(ticket: str, agent=None) -> dict:
    """Run one ticket through the agent and return a structured result."""
    agent = agent or build_agent()
    state = agent.invoke({"messages": [("user", ticket)]})
    msgs = state["messages"]
    category, team, looked_up = _extract(msgs)
    return {
        "reply": _final_reply(msgs),
        "category": category,
        "team": team,
        "looked_up_order": looked_up,
        "escalated": bool(team) and team != "none",
        "num_messages": len(msgs),
    }


if __name__ == "__main__":
    ticket = " ".join(sys.argv[1:]) or "My order #4471 arrived smashed, I want a refund"
    print(f"TICKET: {ticket}\n" + "-" * 60)
    out = run_triage(ticket)
    for k in ("category", "team", "escalated", "looked_up_order", "num_messages"):
        print(f"{k:>17}: {out[k]}")
    print("-" * 60 + f"\nREPLY: {out['reply']}")
