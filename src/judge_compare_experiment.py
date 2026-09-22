"""Run one judge as its own LangSmith experiment on triage-tickets,
grading the SAME frozen agent replies. Lands it in the Experiments tab
next to the baseline runs, with all four feedback columns populated --
category_match / escalation_match / tool_discipline are recomputed from
the frozen (agent_*, ref_*) fields using the SAME evaluator functions the
baseline used, so those three columns are apples-to-apples, not re-derived
by the judge. Only llm_judge is judge-specific.

  python src/judge_compare_experiment.py            # openai only (default)
  python src/judge_compare_experiment.py ollama      # gemma only
  python src/judge_compare_experiment.py all         # both

  --reps N   grade each frozen reply N times (default 2) -> N*14 runs,
             so llm_judge's own variance shows up in ONE experiment.
"""
import argparse
import json
import os
import sys

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))
load_dotenv()
from langsmith import Client  # noqa: E402
from judges import build_judges, judge_for_model  # noqa: E402
from evaluators import category_match, escalation_match, tool_discipline  # noqa: E402

FROZEN = os.path.join(os.path.dirname(__file__), "..", "data", "frozen_replies.json")
DATASET_NAME = os.getenv("LANGSMITH_DATASET", "triage-tickets")


def llm_judge_from_output(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Read the verdict target() already computed -- no new LLM call here.

    The grading call has to happen INSIDE target(), not in a separate
    evaluator: LangSmith traces an evaluator's LLM calls to a different
    project (its 'evaluators' project), so their tokens never reach this
    experiment's own Tokens graph. Doing the grading in target() makes it
    a child span of the root run, which is what the graph actually reads.
    """
    if not outputs.get("judge_ok"):
        return {"key": "llm_judge", "score": 0.0,
                "comment": outputs.get("judge_reason", "judge call failed")}
    # .2f, not .0f: Jev returns fractional positions between levels, and
    # rounding those away would discard exactly what makes it different.
    comment = (f"empathy={outputs['empathy']:.2f} "
               f"completeness={outputs['completeness']:.2f}")
    if outputs.get("confidence") is not None:
        comment += f" confidence={outputs['confidence']:.2f}"
    return {"key": "llm_judge", "score": outputs["llm_score"], "comment": comment}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("judges", nargs="?", default="openai",
                    help="'openai', 'ollama', or 'all'")
    ap.add_argument("--reps", type=int, default=2,
                    help="grade each frozen reply N times (default 2)")
    ap.add_argument("--model", default=None,
                    help="run exactly this OpenAI model id as the judge, "
                         "ignoring the 'judges' positional arg (e.g. gpt-5.6-terra)")
    a = ap.parse_args()

    if not os.path.exists(FROZEN):
        sys.exit("Run src/freeze_replies.py first")
    frozen = {r["ticket"]: r for r in json.load(open(FROZEN))}

    client = Client()

    # The dataset picked up stray examples (raw ChatOpenAI LLM-call shape,
    # not a triage ticket) from outside this script. Rather than delete
    # shared state, filter to only the examples this eval actually needs.
    ds = client.read_dataset(dataset_name=DATASET_NAME)
    all_ex = list(client.list_examples(dataset_id=ds.id))
    clean_examples = [e for e in all_ex
                      if isinstance(e.inputs, dict) and "ticket" in e.inputs]
    stray = len(all_ex) - len(clean_examples)
    if stray:
        print(f"NOTE: skipping {stray} example(s) in '{DATASET_NAME}' that "
              f"aren't triage tickets (not deleted, just excluded here).\n")

    if a.model:
        judges = [judge_for_model(a.model)]
    else:
        judges = (build_judges() if a.judges == "all"
                 else [j for j in build_judges() if j.kind == a.judges])
    if not judges:
        sys.exit(f"No judge of kind {a.judges!r} configured "
                 f"(available kinds: ollama, openai)")
    print(f"running: {', '.join(j.name for j in judges)}  x {a.reps} reps  "
          f"({len(clean_examples)} tickets -> {len(clean_examples)*a.reps} runs/judge)\n")

    for j in judges:
        def target(inputs: dict) -> dict:
            # Frozen reply/routing (fixed facts about that agent run) PLUS
            # the actual judge call, made HERE so its LLM span -- and its
            # token usage -- is a child of this root run, not a detached
            # trace in a separate project. `j` is closed over from the loop;
            # client.evaluate() below runs synchronously before `j` advances,
            # so this is safe despite the classic late-binding closure trap.
            f = frozen[inputs["ticket"]]
            v = j.grade(inputs["ticket"], f["reply"])
            out = {"reply": f["reply"],
                  "category": f["agent_category"],
                  "team": f["agent_team"],
                  "looked_up_order": f["agent_looked_up_order"],
                  "judge_ok": v.ok}
            if v.ok:
                out.update(llm_score=v.score, empathy=v.empathy,
                           completeness=v.completeness,
                           judge_seconds=round(v.seconds, 3),
                           in_tokens=v.in_tokens, out_tokens=v.out_tokens)
                if v.confidence is not None:
                    out["confidence"] = v.confidence
            else:
                out["judge_reason"] = v.reason
            return out

        print(f"--- {j.name} ---")
        client.evaluate(
            target,
            data=clean_examples,   # explicit list -- skips any stray
            evaluators=[category_match, escalation_match, tool_discipline,
                       llm_judge_from_output],
            experiment_prefix=f"judge-{j.name}",
            num_repetitions=a.reps,
            max_concurrency=2,
            metadata={"role": "judge-comparison", "judge_model": j.model,
                      "frozen_replies": True, "reps": a.reps},
        )
        print()

    print(f"Experiment(s) are on the '{DATASET_NAME}' dataset's Experiments tab,")
    print("next to the baseline runs -- all four feedback columns are populated")
    print("and directly comparable (category/escalation/tool_discipline come")
    print("from the frozen agent decision, not from the judge).")


if __name__ == "__main__":
    main()
