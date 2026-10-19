"""Paired bootstrap comparison of two evaluation runs on the same questions.

For each search price, resample questions with replacement and recompute the
utility difference (B - A), giving a 95% interval for how much B beats A.

  python scripts/compare.py results/curve_base_wiki.jsonl results/curve_sft_wiki.jsonl
"""
import json
import random
import sys


def load(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["method"] in ("react", "trained") and "per_q" in r:
            out[r["lam"]] = {qid: (f1, s) for qid, f1, s in r["per_q"]}
    return out


def main(a_path, b_path, iters=5000, seed=0):
    A, B = load(a_path), load(b_path)
    rng = random.Random(seed)
    print(f"A = {a_path}\nB = {b_path}")
    print("lambda  utility A  utility B  B-A     95% CI            P(B>A)")
    for lam in sorted(set(A) & set(B)):
        ids = sorted(set(A[lam]) & set(B[lam]))
        d = [(B[lam][i][0] - lam * B[lam][i][1]) - (A[lam][i][0] - lam * A[lam][i][1]) for i in ids]
        ua = sum(A[lam][i][0] - lam * A[lam][i][1] for i in ids) / len(ids)
        ub = sum(B[lam][i][0] - lam * B[lam][i][1] for i in ids) / len(ids)
        boots = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(iters))
        lo, hi = boots[int(0.025 * iters)], boots[int(0.975 * iters)]
        p = sum(x > 0 for x in boots) / iters
        print(f"{lam:<6}  {ua:.3f}      {ub:.3f}      {ub - ua:+.3f}  [{lo:+.3f}, {hi:+.3f}]  {p:.3f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
