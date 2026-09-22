"""Practice run: ONE Jev Score question against ONE frozen reply.

Test case is deliberately the sharpest disagreement we have. The warranty
reply never answers the question -- it only promises to come back later:

    gemma4-judge   completeness 5/5   (0.750 overall)
    gpt-4o         completeness 2/5   (0.375 overall)
    gpt-5.6-terra  completeness ~1/5  (0.250 overall)

If Jev reads the concrete criteria the way they're written, it should land
near the bottom. That makes this a real test of the rubric, not a smoke test.

  python src/jev_test.py
"""
import json
import os
import sys
import time

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))
load_dotenv()

from typesafe_sdk import Score, TypeSafeClient  # noqa: E402
from typesafe_sdk import TypeSafeAuthenticationError, TypeSafeError  # noqa: E402

FROZEN = os.path.join(os.path.dirname(__file__), "..", "data", "frozen_replies.json")
JEV_MODEL = os.getenv("JEV_MODEL", "jev-latest")

# Levels are ORDERED low -> high and 0-indexed, so this is a 0..4 scale.
# TypeSafe's guidance is concrete situations, never vague degrees ("low /
# medium / high"): the model matches the content against each description
# independently, without seeing the level numbers or its neighbours.
#
# This matters here specifically. Our old free-text rubric said only
# "says what was done and what happens next; no unanswered question" --
# vague enough that gemma handed 5/5 to a reply that answers nothing.
# Level 1 below describes that exact failure, so it can't hide.
COMPLETENESS_LEVELS = [
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


def main() -> None:
    if not os.getenv("TYPESAFE_API_KEY"):
        sys.exit("TYPESAFE_API_KEY is not set in .env -- add it, then re-run.")

    rows = json.load(open(FROZEN))
    row = next((r for r in rows if r["ticket"].startswith("How long is the warranty")),
               None)
    if row is None:
        sys.exit("Expected warranty ticket not found in frozen_replies.json")

    print("TICKET:", row["ticket"])
    print("REPLY :", row["reply"])
    print("-" * 74)

    # STATE = everything a panel of experts would need in front of them.    
    # The ticket is included because completeness is relative to what was
    # asked -- the reply alone can't tell you if a question went unanswered.
    state = {"customer_ticket": row["ticket"], "agent_reply": row["reply"]}

    try:
        with TypeSafeClient() as client:       # reads TYPESAFE_API_KEY from env
            t0 = time.time()
            response = client.system_one(
                state=state,
                questions={
                    "completeness": Score(
                        instructions="How completely does the agent's reply "
                                     "resolve what the customer asked?",
                        criteria=COMPLETENESS_LEVELS,
                    ),
                },
                model=JEV_MODEL,
            )
            secs = time.time() - t0
    except TypeSafeAuthenticationError as e:
        sys.exit(f"Auth failed -- check TYPESAFE_API_KEY: {e}")
    except TypeSafeError as e:
        sys.exit(f"{type(e).__name__}: {e}")

    ans = response.scores["completeness"]
    top = len(COMPLETENESS_LEVELS) - 1

    print(f"score       : {ans.score:.3f}  (0..{top} scale)")
    print(f"normalised  : {ans.score / top:.3f}  (0..1, comparable to our judges)")
    print(f"confidence  : {ans.confidence:.3f}")
    print(f"latency     : {secs * 1000:.0f} ms")
    print("\nprobabilities per level:")
    for lvl, p in sorted(ans.probabilities.items()):
        bar = "#" * int(round(p * 40))
        print(f"  {lvl}  {p:5.3f}  {bar}")
    print("\nlegend:")
    for lvl, desc in sorted(ans.legend.items()):
        print(f"  {lvl}  {desc}")

    print("\n" + "-" * 74)
    print("for reference, on this same reply:")
    print(f"  gemma4-judge   completeness 5/5  -> normalised 1.000")
    print(f"  gpt-4o         completeness 2/5  -> normalised 0.250")
    print(f"  gpt-5.6-terra  completeness ~1/5 -> normalised 0.000-0.250")
    print(f"  jev            completeness      -> normalised {ans.score / top:.3f}")


if __name__ == "__main__":
    main()
