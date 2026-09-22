"""Benchmark judges against each other on frozen agent replies.

  python src/judge_bench.py --reps 3

Because the replies are frozen, every difference below belongs to the judge.
There is no human oracle here, so this measures AGREEMENT and RELIABILITY --
never accuracy. A judge can be perfectly self-consistent and consistently wrong.
"""
import argparse
import json
import os
import statistics as st
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from judges import build_judges  # noqa: E402

FROZEN = os.path.join(os.path.dirname(__file__), "..", "data", "frozen_replies.json")
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "judge_bench.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3,
                    help="grade every reply N times per judge (variance needs N>1)")
    a = ap.parse_args()

    if not os.path.exists(FROZEN):
        sys.exit("No data/frozen_replies.json -- run src/freeze_replies.py first")
    rows = json.load(open(FROZEN))
    judges = build_judges()

    print(f"{len(rows)} frozen replies x {a.reps} reps x {len(judges)} judges "
          f"= {len(rows)*a.reps*len(judges)} calls")
    print("judges:", ", ".join(j.name for j in judges), "\n")
    if len(judges) < 2:
        print("NOTE: only the baseline judge is configured. Set OPENAI_API_KEY")
        print("      in .env to add the second judge.\n")

    results = defaultdict(lambda: defaultdict(list))   # judge -> ticket -> [Verdict]
    for j in judges:
        print(f"--- {j.name} ---")
        fails = 0
        for rep in range(a.reps):
            for i, r in enumerate(rows, 1):
                v = j.grade(r["ticket"], r["reply"])
                results[j.name][r["ticket"]].append(v)
                if not v.ok:
                    fails += 1
            done = (rep + 1) * len(rows)
            print(f"   rep {rep+1}/{a.reps} done ({done} calls, {fails} failures)")
        print()

    # ------------------------------------------------------------- per judge
    print("=" * 74)
    print("PER-JUDGE  (frozen replies, so all variation is the judge)")
    summary = {}
    for j in judges:
        per = results[j.name]
        ok = [v for vs in per.values() for v in vs if v.ok]
        allv = [v for vs in per.values() for v in vs]
        if not ok:
            print(f"   {j.name}: all {len(allv)} calls failed to parse")
            continue
        # reliability = mean per-ticket variance across repeats
        varis = [st.pvariance([v.score for v in vs if v.ok])
                 for vs in per.values() if len([v for v in vs if v.ok]) > 1]
        mean_var = st.mean(varis) if varis else 0.0
        lat = st.mean(v.seconds for v in allv)
        tin = sum(v.in_tokens for v in allv)
        tout = sum(v.out_tokens for v in allv)
        cost = j.cost(tin, tout)
        summary[j.name] = {
            "mean_score": st.mean(v.score for v in ok),
            "mean_per_ticket_variance": mean_var,
            "parse_failures": len(allv) - len(ok),
            "calls": len(allv),
            "mean_latency_s": lat,
            "in_tokens": tin, "out_tokens": tout,
            "total_cost_usd": cost,
        }
        print(f"\n   {j.name}")
        print(f"     mean score        : {st.mean(v.score for v in ok):.4f}")
        print(f"     per-ticket variance: {mean_var:.7f}   (lower = more reliable)")
        print(f"     parse failures    : {len(allv)-len(ok)}/{len(allv)}")
        print(f"     mean latency      : {lat:.2f}s")
        print(f"     tokens            : {tin} in / {tout} out")
        print(f"     cost              : "
              + (f"${cost:.4f}" if cost is not None else "unknown (no price set)"))

    # ------------------------------------------------------------- agreement
    if len(judges) >= 2:
        a_, b_ = judges[0].name, judges[1].name
        print("\n" + "=" * 74)
        print(f"AGREEMENT  {a_}  vs  {b_}")
        diffs, exact, rows_out = [], 0, []
        for r in rows:
            va = [v.score for v in results[a_][r["ticket"]] if v.ok]
            vb = [v.score for v in results[b_][r["ticket"]] if v.ok]
            if not va or not vb:
                continue
            ma, mb = st.mean(va), st.mean(vb)
            diffs.append(abs(ma - mb))
            exact += abs(ma - mb) < 1e-9
            rows_out.append((r["ticket"], ma, mb, ma - mb))
        if diffs:
            print(f"   mean |difference| : {st.mean(diffs):.4f}")
            print(f"   exact agreement   : {exact}/{len(diffs)}")
            print(f"\n   {'ticket':<44}{a_[:9]:>9}{b_[:9]:>9}{'delta':>8}")
            for t, ma, mb, d in sorted(rows_out, key=lambda x: -abs(x[3])):
                print(f"   {t[:44]:<44}{ma:>9.3f}{mb:>9.3f}{d:>+8.3f}")
        summary["agreement"] = {
            "pair": [a_, b_],
            "mean_abs_diff": st.mean(diffs) if diffs else None,
            "exact_agreement": f"{exact}/{len(diffs)}" if diffs else None,
        }

    json.dump({"reps": a.reps, "n_replies": len(rows), "summary": summary},
              open(OUT, "w"), indent=2, default=str)
    print(f"\n-> {os.path.relpath(OUT)}")
    print("\nNo human labels here: this is agreement and reliability, not accuracy.")


if __name__ == "__main__":
    main()
