"""Plot search count, utility and F1 against the search price for each model.

  python scripts/plot_curve.py --runs "untrained=results/curve_base_wiki.jsonl" \
      "SFT=results/curve_sft_wiki.jsonl" --out results/curve_wiki.png
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    rows = [r for r in rows if r["method"] in ("react", "trained")]
    return sorted(rows, key=lambda r: r["lam"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="label=path.jsonl")
    ap.add_argument("--no_search", default=None, help="jsonl with a no_search row, drawn as a reference")
    ap.add_argument("--out", default="results/curve_wiki.png")
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for spec in args.runs:
        label, path = spec.split("=", 1)
        rows = load(path)
        lam = [r["lam"] for r in rows]
        axes[0].plot(lam, [r["searches"] for r in rows], "o-", label=label)
        axes[1].plot(lam, [r["utility"] for r in rows], "o-", label=label)
        axes[2].plot([r["searches"] for r in rows], [r["f1"] for r in rows], "o-", label=label)
        for r in rows:
            axes[2].annotate(f"λ={r['lam']:g}", (r["searches"], r["f1"]), fontsize=7,
                             xytext=(3, 3), textcoords="offset points")
    if args.no_search:
        ns = [json.loads(l) for l in open(args.no_search, encoding="utf-8")]
        ns = [r for r in ns if r["method"] == "no_search"][0]
        axes[1].axhline(ns["utility"], ls="--", c="gray", lw=1, label="never search")
        axes[2].plot([0], [ns["f1"]], "s", c="gray", label="never search")
    axes[0].set(xlabel="search price λ", ylabel="mean searches per question", title="Searches follow the price")
    axes[1].set(xlabel="search price λ", ylabel="utility = F1 − λ·searches", title="Utility")
    axes[2].set(xlabel="mean searches per question", ylabel="F1", title="F1 vs searches")
    for a in axes:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.suptitle("HotpotQA dev[:200], 5.2M-passage Wikipedia BM25, Qwen3-1.7B, max 3 searches, greedy",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print("saved", args.out)


if __name__ == "__main__":
    main()
