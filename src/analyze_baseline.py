"""Turn repeated experiment runs into a baseline with error bars.

Answers three questions, in order of how much they matter:
  1. Do the agent's ROUTING decisions ever flip across repeats?
  2. How much does the judge score vary per example?
  3. What is the aggregate noise floor -- the minimum detectable effect?
"""
import json
import os
import re
import statistics as st
import sys
from collections import defaultdict

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))
load_dotenv()
from langsmith import Client  # noqa: E402

DET = ["category_match", "escalation_match", "tool_discipline"]
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "baseline.json")


def collect(prefixes):
    """Gather every finished run across the named experiments."""
    c = Client()
    samples = []
    for p in c.list_projects():
        if not any(p.name.startswith(pre) for pre in prefixes):
            continue
        for r in c.list_runs(project_name=p.name, is_root=True):
            if not r.end_time:
                continue
            fb = {f.key: f for f in c.list_feedback(run_ids=[r.id])}
            if "llm_judge" not in fb:
                continue
            m = re.search(r"empathy=(\d+)\s+completeness=(\d+)",
                          fb["llm_judge"].comment or "")
            samples.append({
                "exp": p.name,
                "ticket": (r.inputs.get("inputs") or r.inputs).get("ticket", ""),
                "start": r.start_time.isoformat(),
                "judge": fb["llm_judge"].score,
                "empathy": int(m.group(1)) if m else None,
                "completeness": int(m.group(2)) if m else None,
                "det": {k: (fb[k].score if k in fb else None) for k in DET},
                "reply": (r.outputs or {}).get("reply", ""),
                "category": (r.outputs or {}).get("category"),
                "team": (r.outputs or {}).get("team"),
            })
    return samples


def main():
    prefixes = sys.argv[1:] or ["baseline-A", "baseline-B"]
    s = collect(prefixes)
    if not s:
        sys.exit("No finished runs found for: " + ", ".join(prefixes))

    exps = sorted({x["exp"] for x in s})
    print(f"{len(s)} samples across {len(exps)} experiments: {', '.join(exps)}\n")

    # --- 1. routing stability -------------------------------------------
    print("=" * 74)
    print("1. ROUTING STABILITY  (does the agent ever decide differently?)")
    flips = [x for x in s if any(v != 1.0 for v in x["det"].values() if v is not None)]
    for k in DET:
        vals = [x["det"][k] for x in s if x["det"][k] is not None]
        print(f"   {k:>17}: {100*sum(vals)/len(vals):6.1f}%   ({len(vals)} runs)")
    if flips:
        print(f"\n   !! {len(flips)} run(s) disagreed with the reference:")
        for x in flips[:8]:
            bad = [k for k, v in x["det"].items() if v != 1.0]
            print(f"      {x['ticket'][:46]:<46} {bad} cat={x['category']} team={x['team']}")
    else:
        print("\n   No flips. Routing is stable; only the wording varies.")

    # --- 2. per-example variance ----------------------------------------
    print("\n" + "=" * 74)
    print("2. PER-EXAMPLE JUDGE VARIANCE")
    by = defaultdict(list)
    for x in s:
        by[x["ticket"]].append(x)
    print(f"   {'ticket':<44}{'n':>3}{'mean':>7}{'sd':>7}{'min':>6}{'max':>6}")
    per_ex = {}
    for t, xs in sorted(by.items(), key=lambda kv: -(
            st.pstdev([y["judge"] for y in kv[1]]) if len(kv[1]) > 1 else 0)):
        js = [y["judge"] for y in xs]
        sd = st.pstdev(js) if len(js) > 1 else 0.0
        per_ex[t] = {"n": len(js), "mean": st.mean(js), "sd": sd,
                     "min": min(js), "max": max(js),
                     "unique_replies": len({y["reply"].strip() for y in xs})}
        print(f"   {t[:44]:<44}{len(js):>3}{st.mean(js):>7.3f}{sd:>7.3f}{min(js):>6.3f}{max(js):>6.3f}")

    unstable = [t for t, v in per_ex.items() if v["sd"] > 0]
    print(f"\n   {len(unstable)}/{len(per_ex)} tickets vary at all across repeats")
    wording = sum(1 for v in per_ex.values() if v["unique_replies"] > 1)
    print(f"   {wording}/{len(per_ex)} tickets produced more than one distinct reply")

    # --- 3. aggregate noise floor ---------------------------------------
    print("\n" + "=" * 74)
    print("3. AGGREGATE NOISE FLOOR")
    # each (experiment, rep-index) is one virtual run of the whole dataset
    virtual = defaultdict(list)
    for t, xs in by.items():
        for i, x in enumerate(sorted(xs, key=lambda y: y["start"])):
            virtual[(x["exp"], i)].append(x["judge"])
    # Only complete passes are comparable -- a pass missing a ticket has a
    # different denominator, which looks like variance but is a gap.
    full = max(len(v) for v in virtual.values())
    dropped = [k for k, v in virtual.items() if len(v) != full]
    aggs = {k: st.mean(v) for k, v in sorted(virtual.items()) if len(v) == full}
    if dropped:
        print(f"   (skipped {len(dropped)} incomplete pass(es); "
              f"complete passes cover {full} tickets)")
    for (exp, i), v in sorted(aggs.items()):
        print(f"   {exp:<26} rep {i}: {100*v:5.2f}%")
    vals = list(aggs.values())
    if len(vals) > 1:
        spread = (max(vals) - min(vals)) * 100
        sd = st.pstdev(vals) * 100
        print(f"\n   mean {100*st.mean(vals):.2f}%   sd {sd:.2f}pp   spread {spread:.2f}pp")
        print(f"   -> a prompt change must move the score by MORE than ~{max(spread, 2*sd):.1f}pp"
              f"\n      before it means anything.")
        mde = max(spread, 2 * sd)
    else:
        mde = None

    json.dump({"samples": len(s), "experiments": exps,
               "routing_flips": len(flips),
               "per_example": per_ex,
               "virtual_runs": {f"{k[0]}#{k[1]}": v for k, v in aggs.items()},
               "aggregate_mean": st.mean(vals) if vals else None,
               "aggregate_sd_pp": st.pstdev(vals) * 100 if len(vals) > 1 else None,
               "min_detectable_effect_pp": mde},
              open(OUT, "w"), indent=2)
    print(f"\n   -> {os.path.relpath(OUT)}")


if __name__ == "__main__":
    main()
