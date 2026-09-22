"""Snapshot one canonical agent reply per ticket.

The agent is not reproducible at temperature 0, so a judge comparison has to
hold its output still: every judge must grade the exact same text. This writes
that frozen set once; after it exists, judge scores are the only moving part.
"""
import json
import os
import sys
from collections import defaultdict

from dotenv import load_dotenv

load_dotenv()
from langsmith import Client  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "frozen_replies.json")
DATA = os.path.join(os.path.dirname(__file__), "..", "data", "dataset.json")


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else "baseline-"
    refs = {d["ticket"]: d for d in json.load(open(DATA))}
    c = Client()

    by = defaultdict(list)
    for p in c.list_projects():
        if not p.name.startswith(prefix):
            continue
        for r in c.list_runs(project_name=p.name, is_root=True):
            if not r.end_time:
                continue
            tk = (r.inputs.get("inputs") or r.inputs).get("ticket", "")
            reply = (r.outputs or {}).get("reply", "").strip()
            if tk and reply:
                by[tk].append({"reply": reply, "start": r.start_time.isoformat(),
                               "exp": p.name,
                               "category": (r.outputs or {}).get("category"),
                               "team": (r.outputs or {}).get("team"),
                               "looked_up_order":
                                   (r.outputs or {}).get("looked_up_order")})

    if not by:
        sys.exit(f"No finished runs found under prefix {prefix!r}")

    frozen, multi = [], 0
    for tk, cands in by.items():
        cands.sort(key=lambda x: x["start"])          # earliest = canonical
        pick = cands[0]
        uniq = {c["reply"] for c in cands}
        multi += len(uniq) > 1
        frozen.append({
            "ticket": tk,
            "reply": pick["reply"],
            "agent_category": pick["category"],
            "agent_team": pick["team"],
            "agent_looked_up_order": pick["looked_up_order"],
            "ref_category": refs.get(tk, {}).get("category"),
            "ref_team": refs.get(tk, {}).get("team"),
            "ref_needs_lookup": refs.get(tk, {}).get("needs_lookup"),
            "source_experiment": pick["exp"],
            "n_candidates": len(cands),
            "n_distinct_replies": len(uniq),
        })

    frozen.sort(key=lambda x: x["ticket"])
    json.dump(frozen, open(OUT, "w"), indent=2)
    print(f"Froze {len(frozen)} replies -> {os.path.relpath(OUT)}")
    print(f"  {multi}/{len(frozen)} tickets had more than one distinct reply "
          f"across runs (earliest kept)")
    print(f"  candidates per ticket: "
          f"{sorted({f['n_candidates'] for f in frozen})}")


if __name__ == "__main__":
    main()
