"""Feature-group ablations (Table VIII; all three scopes in Table S7).

Each news feature group is removed from the dataset and the model is retrained from scratch with the same
seeds. The ablated model (arm no-<group>) is then compared with the full model (arm news) on calibrated
p95 Winkler loss, with the paired machinery of paired_stats.py. Reported per group and scope: the mean
daily loss increase as a percentage of the full model's loss, and the Newey-West p-value with a
Bonferroni correction over all group x scope comparisons.

Workflow:
    python src/ablation.py --commands          # prints the dataset and training commands of every group
    ... run them ...
    python src/ablation.py                     # Table VIII
"""
import argparse
import os

import pandas as pd

import config as C
from news_features import GROUPS
from paired_stats import SCOPES, collect, summarize

LABELS = {"META": "Availability/attribution", "FLOW": "News flow", "EMBED": "Title-embedding pipeline",
          "NOVELTY": "Semantic change", "EVENT": "Event content", "SENT": "Sentiment"}


def arm_of(group):
    return f"no-{group.lower()}"


def print_commands(seeds):
    for g in LABELS:
        ds = os.path.join("artifacts", f"dataset_{arm_of(g)}.pt")
        print(f"python src/dataset.py --drop-group {g} --out {ds}")
        for s in seeds:
            print(f"python src/train.py --data {ds} --arm {arm_of(g)} --seed {s}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--commands", action="store_true", help="print the dataset/training commands")
    ap.add_argument("--seeds", default=",".join(map(str, C.SEEDS)))
    ap.add_argument("--preds-dir", default=C.PREDS_DIR)
    ap.add_argument("--news-dir", default=C.NEWS_DIR)
    ap.add_argument("--out-dir", default=C.RESULTS_DIR)
    ap.add_argument("--workers", type=int, default=max((os.cpu_count() or 2) - 1, 1))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.commands:
        return print_commands(seeds)

    rows = []
    for g, label in LABELS.items():
        df = collect([arm_of(g), "news"], seeds, "cal", args.preds_dir, args.news_dir, args.workers)
        for scope, sel in SCOPES:
            r = summarize(df[sel(df)])
            rows.append({"group": g, "removed": label, "scope": scope, "increase%": r["change%"],
                         "diff": r["diff"], "p": r["p"]})
    T = pd.DataFrame(rows)
    T["p_bonferroni"] = (T["p"] * len(T)).clip(upper=1.0)
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "ablation.csv")
    T.to_csv(out, index=False)

    wide = T.pivot(index="removed", columns="scope", values="increase%").reindex(list(LABELS.values()))
    sig = T.pivot(index="removed", columns="scope", values="p_bonferroni").reindex(list(LABELS.values()))
    print(f"\n[ablation] increase in calibrated p95 Winkler after removing each group (seeds {seeds})")
    print(f"{'removed component':<26} {'All':>9} {'Event':>9} {'Non-event':>10}")
    for name in wide.index:
        cells = [f"{wide.loc[name, s]:+8.2f}{'*' if sig.loc[name, s] < 0.05 else ' '}" for s in ("All", "Event", "Non-event")]
        print(f"{name:<26} " + " ".join(cells))
    print(f"\n* significant after Bonferroni correction over {len(T)} comparisons   -> {out}")


if __name__ == "__main__":
    main()
