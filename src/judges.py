"""Pluggable judges scoring the SAME frozen replies.

Both backends get byte-identical prompts and identical parsing, so any
difference in their scores is the judge, not the harness.
"""
import json
import os
import re
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

# The single shared rubric. Neither backend may vary this text.
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


def parse_verdict(text: str) -> dict | None:
    """Pull the JSON verdict out, tolerating fenced or chatty output."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        e, c = float(d["empathy"]), float(d["completeness"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    if not (1 <= e <= 5 and 1 <= c <= 5):
        return None
    return {"empathy": e, "completeness": c,
            "reason": str(d.get("reason", ""))[:200]}


def to_score(empathy: float, completeness: float) -> float:
    """1-5 per axis -> mean normalised to 0-1 (1 is the floor, so 1 -> 0)."""
    return (((empathy + completeness) / 2.0) - 1.0) / 4.0


@dataclass
class Verdict:
    ok: bool
    score: float | None = None
    empathy: float | None = None
    completeness: float | None = None
    reason: str = ""
    seconds: float = 0.0
    in_tokens: int = 0
    out_tokens: int = 0
    raw: str = ""
    confidence: float | None = None   # Jev only; text judges give no calibration


# --- Jev rubric -------------------------------------------------------------
# Jev scores against ORDERED level descriptions rather than a free-text prompt,
# and TypeSafe's guidance is concrete situations, never vague degrees: it
# matches the state against each description independently, without seeing the
# level numbers or its neighbours.
#
# Levels are 0-indexed (0..4). grade() adds 1 so the result lands on the same
# 1..5 scale the text judges use, which means to_score() -- and therefore the
# llm_judge column in LangSmith -- stays directly comparable across all four.
JEV_EMPATHY_LEVELS = [
    "Purely transactional; no acknowledgement that the customer is affected",
    "Generic opener that restates the request without recognising any frustration",
    "Acknowledges the situation, but in standard formulaic language",
    "Names the specific inconvenience this customer is experiencing",
    "Apologises for the specific problem and recognises its impact in the "
    "customer's own terms",
]

JEV_COMPLETENESS_LEVELS = [
    "Acknowledges the problem but answers nothing and commits to no action",
    "Promises to look into it or get back to the customer, with no answer "
    "and no timeframe",
    "States that an action was taken, but does not say who now owns it or "
    "when the customer will hear back",
    "States the action taken and names the team or next step, but gives no "
    "specific timeframe",
    "Answers the question or states the action taken, names who now owns it, "
    "and sets a clear expectation for what happens next",
]


@dataclass
class Judge:
    name: str
    kind: str                       # "ollama" | "openai" | "jev"
    model: str
    price_in: float | None = None   # USD per 1M input tokens, if known
    price_out: float | None = None
    _llm: object = field(default=None, repr=False)

    def _client(self):
        if self._llm is None:
            if self.kind == "ollama":
                from langchain_ollama import ChatOllama
                self._llm = ChatOllama(model=self.model, temperature=0)
            elif self.kind == "openai":
                from langchain_openai import ChatOpenAI
                self._llm = ChatOpenAI(model=self.model, temperature=0)
            elif self.kind == "jev":
                from typesafe_sdk import TypeSafeClient
                self._llm = TypeSafeClient()   # reads TYPESAFE_API_KEY from env
            else:
                raise ValueError(f"unknown judge kind {self.kind!r}")
        return self._llm

    def grade(self, ticket: str, reply: str) -> Verdict:
        if self.kind == "jev":
            return self._grade_jev(ticket, reply)
        return self._grade_text(ticket, reply)

    def _grade_jev(self, ticket: str, reply: str) -> Verdict:
        """Two one-dimensional Score questions against the same state.

        Kept as separate questions deliberately: TypeSafe warns that
        multi-aspect questions split the model's attention and lower
        confidence, and our rubric already treats the two axes separately.
        """
        from typesafe_sdk import Score, TypeSafeError

        state = {"customer_ticket": ticket, "agent_reply": reply}
        t0 = time.time()
        try:
            resp = self._client().system_one(
                state=state,
                questions={
                    "empathy": Score(
                        instructions="How well does the agent's reply "
                                     "acknowledge this customer's situation?",
                        criteria=JEV_EMPATHY_LEVELS,
                    ),
                    "completeness": Score(
                        instructions="How completely does the agent's reply "
                                     "resolve what the customer asked?",
                        criteria=JEV_COMPLETENESS_LEVELS,
                    ),
                },
                model=self.model,
            )
        except TypeSafeError as e:                  # noqa: BLE001
            return Verdict(ok=False, seconds=time.time() - t0,
                           reason=f"{type(e).__name__}: {e}"[:200])
        except Exception as e:                      # noqa: BLE001
            return Verdict(ok=False, seconds=time.time() - t0,
                           reason=f"{type(e).__name__}: {e}"[:200])
        secs = time.time() - t0

        e_ans, c_ans = resp.scores["empathy"], resp.scores["completeness"]
        # 0..4 -> 1..5, so to_score() and the llm_judge column match the
        # text judges exactly. (score/4 == ((score+1)-1)/4, so the
        # normalisation is unchanged -- only the reported scale shifts.)
        empathy = e_ans.score + 1.0
        completeness = c_ans.score + 1.0
        conf = (e_ans.confidence + c_ans.confidence) / 2.0
        usage = getattr(resp, "usage", None)
        return Verdict(
            ok=True,
            score=to_score(empathy, completeness),
            empathy=empathy, completeness=completeness,
            confidence=conf, seconds=secs,
            reason=(f"empathy conf={e_ans.confidence:.2f} "
                    f"completeness conf={c_ans.confidence:.2f}"),
            raw=f"empathy_probs={e_ans.probabilities} "
                f"completeness_probs={c_ans.probabilities}"[:200],
            in_tokens=getattr(usage, "input_tokens", 0) or 0,
            out_tokens=getattr(usage, "output_tokens", 0) or 0,
        )

    def _grade_text(self, ticket: str, reply: str) -> Verdict:
        prompt = JUDGE_PROMPT.format(ticket=ticket, reply=reply)
        t0 = time.time()
        try:
            msg = self._client().invoke(prompt)
        except Exception as e:                      # noqa: BLE001
            return Verdict(ok=False, seconds=time.time() - t0,
                           reason=f"{type(e).__name__}: {e}"[:200])
        secs = time.time() - t0
        raw = str(msg.content)
        usage = getattr(msg, "usage_metadata", None) or {}
        v = parse_verdict(raw)
        if v is None:
            return Verdict(ok=False, seconds=secs, raw=raw[:200],
                           reason="unparseable verdict",
                           in_tokens=usage.get("input_tokens", 0),
                           out_tokens=usage.get("output_tokens", 0))
        return Verdict(ok=True, score=to_score(v["empathy"], v["completeness"]),
                       empathy=v["empathy"], completeness=v["completeness"],
                       reason=v["reason"], seconds=secs, raw=raw[:200],
                       in_tokens=usage.get("input_tokens", 0),
                       out_tokens=usage.get("output_tokens", 0))

    def cost(self, in_tok: int, out_tok: int) -> float | None:
        if self.price_in is None or self.price_out is None:
            return None
        return (in_tok * self.price_in + out_tok * self.price_out) / 1e6


def build_judges() -> list[Judge]:
    """Baseline first; OpenAI only when a key is actually present."""
    js = [Judge(name="gemma4-judge", kind="ollama",
                model=os.getenv("JUDGE_MODEL", "gemma4-judge:latest"),
                price_in=0.0, price_out=0.0)]      # local: no marginal cost
    if os.getenv("OPENAI_API_KEY"):
        js.append(Judge(name=os.getenv("OPENAI_JUDGE_MODEL", "gpt-4o-mini"),
                        kind="openai",
                        model=os.getenv("OPENAI_JUDGE_MODEL", "gpt-4o-mini")))
    return js


def judge_for_model(model: str, kind: str | None = None) -> Judge:
    """One explicit judge by model id, bypassing the env-var defaults --
    for ad-hoc comparisons against whichever model you name on the CLI.

    Kind is inferred from the model id when not given, so `--model jev-latest`
    routes to the TypeSafe SDK rather than being treated as an OpenAI id.
    """
    if kind is None:
        kind = "jev" if model.startswith("jev") else "openai"
    return Judge(name=model, kind=kind, model=model)
