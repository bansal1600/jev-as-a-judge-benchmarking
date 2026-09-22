"""Evaluate the triage agent.

  python src/run_eval.py              # local scorecard, no API key needed
  python src/run_eval.py --langsmith  # upload dataset + run a tracked experiment
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))
load_dotenv()

from agent import build_agent, run_triage  # noqa: E402
from evaluators import ALL_EVALUATORS  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "dataset.json")
DATASET_NAME = os.getenv("LANGSMITH_DATASET", "triage-tickets")


def load_dataset() -> list[dict]:
    with open(DATA) as f:
        return json.load(f)


def _split(ex: dict) -> tuple[dict, dict]:
    """Split a raw example into (inputs, reference_outputs)."""
    return ({"ticket": ex["ticket"]},
            {k: ex[k] for k in ("category", "team", "needs_lookup")})


# ------------------------------------------------------------------ local mode


def run_local() -> None:
    rows = load_dataset()
    agent = build_agent()
    totals: dict[str, list[float]] = defaultdict(list)
    results = []

    print(f"Running {len(rows)} examples locally (no LangSmith)\n" + "=" * 78)
    for i, ex in enumerate(rows, 1):
        inputs, refs = _split(ex)
        t0 = time.time()
        outputs = run_triage(inputs["ticket"], agent=agent)
        secs = time.time() - t0

        scores, comment = {}, ""
        for ev in ALL_EVALUATORS:
            r = ev(inputs, outputs, refs)
            scores[r["key"]] = r["score"]
            totals[r["key"]].append(r["score"])
            if r.get("comment"):
                comment = r["comment"]

        flags = "".join("." if scores[k] == 1.0 else "X"
                        for k in ("category_match", "escalation_match", "tool_discipline"))
        print(f"[{i:>2}/{len(rows)}] {flags} judge={scores['llm_judge']:.2f} "
              f"{secs:>5.1f}s | {inputs['ticket'][:52]}")
        if flags != "...":
            print(f"         got category={outputs['category']} team={outputs['team']} "
                  f"lookup={outputs['looked_up_order']}")
            print(f"         want category={refs['category']} team={refs['team']} "
                  f"lookup={refs['needs_lookup']}")

        results.append({"inputs": inputs, "reference": refs,
                        "outputs": outputs, "scores": scores, "comment": comment,
                        "seconds": round(secs, 1)})

    print("=" * 78 + "\nSCORECARD")
    for k, vals in totals.items():
        pct = 100 * sum(vals) / len(vals)
        bar = "#" * int(pct / 5)
        print(f"  {k:>17}: {pct:5.1f}%  {bar}")

    out = os.path.join(os.path.dirname(__file__), "..", "data", "results_local.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nPer-example detail -> {os.path.relpath(out)}")


# -------------------------------------------------------------- langsmith mode


def run_langsmith(reps: int = 1, concurrency: int = 1,
                  prefix: str = "triage-gemma4") -> None:
    from langsmith import Client

    if not os.getenv("LANGSMITH_API_KEY", "").startswith("lsv2"):
        sys.exit("No valid LANGSMITH_API_KEY in .env -- get one at smith.langchain.com")
    os.environ["LANGSMITH_TRACING"] = "true"

    client = Client()
    rows = load_dataset()

    if client.has_dataset(dataset_name=DATASET_NAME):
        ds = client.read_dataset(dataset_name=DATASET_NAME)
        print(f"Reusing dataset '{DATASET_NAME}' ({ds.example_count} examples)")
    else:
        ds = client.create_dataset(DATASET_NAME,
                                   description="Customer-support triage tickets")
        client.create_examples(
            dataset_id=ds.id,
            examples=[{"inputs": i, "outputs": o}
                      for i, o in (_split(ex) for ex in rows)],
        )
        print(f"Created dataset '{DATASET_NAME}' with {len(rows)} examples")

    agent = build_agent()

    def target(inputs: dict) -> dict:
        return run_triage(inputs["ticket"], agent=agent)

    print(f"Running {len(rows)} examples x {reps} reps "
          f"= {len(rows)*reps} runs, concurrency {concurrency} ...")
    res = client.evaluate(
        target,
        data=DATASET_NAME,
        evaluators=ALL_EVALUATORS,
        experiment_prefix=prefix,
        num_repetitions=reps,       # same example N times -> variance, not a point
        max_concurrency=concurrency,
        metadata={"agent_model": os.getenv("AGENT_MODEL"),
                  "judge_model": os.getenv("JUDGE_MODEL"),
                  "reps": reps, "concurrency": concurrency},
    )
    print("\nDone. Open the experiment in LangSmith:")
    print(f"  https://smith.langchain.com/  ->  Datasets  ->  {DATASET_NAME}")
    try:
        print(f"  {res._manager._experiment.url}")   # best-effort direct link
    except Exception:
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--langsmith", action="store_true",
                    help="run as a tracked LangSmith experiment")
    ap.add_argument("--reps", type=int, default=1,
                    help="run each example N times (needs --langsmith)")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="parallel runs against the local Ollama server")
    ap.add_argument("--prefix", default="triage-gemma4",
                    help="experiment name prefix")
    a = ap.parse_args()
    if a.langsmith:
        run_langsmith(reps=a.reps, concurrency=a.concurrency, prefix=a.prefix)
    else:
        run_local()
