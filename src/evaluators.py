"""Evaluators scoring the triage agent.

Signature is LangSmith's modern one -- (inputs, outputs, reference_outputs) --
so the same functions run locally and inside client.evaluate() unchanged.
Each returns {"key": ..., "score": ...} with score in [0, 1].
"""
import json
import os
import re

from langchain_ollama import ChatOllama

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemma4-judge:latest")

# ---------------------------------------------------------------- deterministic


def category_match(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Did the agent pick the right category from the taxonomy?"""
def _missing(key: str, *sides) -> dict | None:
    """Absence is not agreement.

    These evaluators used to coerce a missing field to "" (or False) on both
    sides and score a perfect 1.0 -- so a pipeline that silently stopped
    emitting `category` would report 100%, indistinguishable from a real pass.
    Score 0 and say which side was empty instead.
    """
    empty = [n for n, v in sides if v is None or (isinstance(v, str) and not v.strip())]
    if not empty:
        return None
    return {"key": key, "score": 0.0,
            "comment": f"missing field(s): {', '.join(empty)} -- not a pass"}


def category_match(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Did the agent pick the right category from the taxonomy?"""
    got, want = outputs.get("category"), reference_outputs.get("category")
    bad = _missing("category_match", ("outputs.category", got),
                   ("reference.category", want))
    if bad:
        return bad
    return {"key": "category_match",
            "score": float(got.strip().lower() == want.strip().lower())}


def escalation_match(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Did it route to the correct specialist team (or correctly to none)?"""
    got, want = outputs.get("team"), reference_outputs.get("team")
    bad = _missing("escalation_match", ("outputs.team", got),
                   ("reference.team", want))
    if bad:
        return bad
    return {"key": "escalation_match",
            "score": float(got.strip().lower() == want.strip().lower())}


def tool_discipline(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Did it look up the order exactly when an order number was present?

    Catches both failure directions: skipping a lookup it needed, and
    hallucinating a lookup for a ticket with no order number.
    """
    needed = reference_outputs.get("needs_lookup")
    did = outputs.get("looked_up_order")
    # Explicit None check, not falsiness -- False is a valid answer here,
    # so `or`-coercion would make a missing field look like a legitimate "no".
    bad = _missing("tool_discipline", ("outputs.looked_up_order", did),
                   ("reference.needs_lookup", needed))
    if bad:
        return bad
    return {"key": "tool_discipline", "score": float(bool(needed) == bool(did))}


# ------------------------------------------------------------------- llm judge

JUDGE_PROMPT = """You are grading a customer-support reply. Be strict but fair.

CUSTOMER TICKET:
{ticket}

AGENT REPLY:
{reply}

Score the reply 1-5 on each:
- empathy: acknowledges the customer's frustration in a warm, human tone
- completeness: says what was done and what happens next; no unanswered question

Respond with ONLY a JSON object, no other text:
{{"empathy": <1-5>, "completeness": <1-5>, "reason": "<one short sentence>"}}"""

_judge_llm = None


def _get_judge():
    global _judge_llm
    if _judge_llm is None:
        _judge_llm = ChatOllama(model=JUDGE_MODEL, temperature=0)
    return _judge_llm


def _parse_scores(text: str) -> dict | None:
    """Pull the JSON verdict out, tolerating fenced or chatty output."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        return {
            "empathy": float(d["empathy"]),
            "completeness": float(d["completeness"]),
            "reason": str(d.get("reason", ""))[:200],
        }
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def llm_judge(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """LLM-as-a-judge on tone and completeness of the customer-facing reply."""
    reply = (outputs.get("reply") or "").strip()
    if not reply:
        return {"key": "llm_judge", "score": 0.0, "comment": "empty reply"}

    prompt = JUDGE_PROMPT.format(ticket=inputs.get("ticket", ""), reply=reply)
    raw = _get_judge().invoke(prompt).content
    scores = _parse_scores(str(raw))
    if scores is None:
        # A judge that can't produce parseable JSON must not silently pass.
        return {"key": "llm_judge", "score": 0.0,
                "comment": f"unparseable verdict: {str(raw)[:120]}"}

    # 1-5 each -> mean normalised to 0-1
    mean = (scores["empathy"] + scores["completeness"]) / 2.0
    return {
        "key": "llm_judge",
        "score": (mean - 1.0) / 4.0,
        "comment": f"empathy={scores['empathy']:.0f} "
                   f"completeness={scores['completeness']:.0f} :: {scores['reason']}",
    }


ALL_EVALUATORS = [category_match, escalation_match, tool_discipline, llm_judge]
