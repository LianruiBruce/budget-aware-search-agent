"""Offline diagnostic: is there anything for a cost-aware search policy to learn?

For every question we force the agent to search exactly k times (k in --ks),
writing its own queries with greedy decoding, and then force an answer. That
gives F1(q, k) for every question and search count. For each search price
lambda we then compare three policies, all scored by F1 - lambda * k:

  fixed    the best single k for all questions
  by_type  the best k per HotpotQA question type (bridge / comparison), a
           crude stand-in for what a policy could realistically learn
  oracle   the best k for each individual question (an upper bound; it also
           picks up per-question luck, so it overstates what is learnable)

If oracle - fixed is small, a learned policy cannot beat the best fixed
strategy by much, and RL in this setting has little to learn.

Usage:
  python scripts/diagnose.py --corpus small --n 300 --out results/diag_small
"""
import argparse
import json
import os
import random
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from basa.data import build  # noqa: E402
from basa.retriever import BM25Retriever  # noqa: E402
from basa.reward import exact_match, f1_score  # noqa: E402
from basa.rollout import SEARCH_RE, rollout  # noqa: E402


def load_retriever(name, corpus):
    if name == "small":
        return BM25Retriever(corpus)
    if name == "wiki":
        from basa.wiki import load_wiki_retriever
        return load_wiki_retriever()
    raise ValueError(name)


def bootstrap_ci(xs, iters=2000, seed=0):
    rng = random.Random(seed)
    n = len(xs)
    means = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return means[int(0.025 * iters)], means[int(0.975 * iters)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--corpus", default="small", choices=["small", "wiki"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--ks", default="0,1,2,3")
    ap.add_argument("--lams", default="0,0.05,0.1,0.2,0.3")
    ap.add_argument("--out", default="results/diag_small")
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",")]
    lams = [float(x) for x in args.lams.split(",")]

    train, _, corpus = build()
    qs = train[:args.n]  # training questions: keep the dev set untouched
    t0 = time.time()
    ret = load_retriever(args.corpus, corpus)
    print(f"retriever ready in {time.time() - t0:.0f}s", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()

    rows = {q["id"]: {"id": q["id"], "type": q["type"], "level": q["level"], "gold": q["answer"],
                      "supporting_titles": q["supporting_titles"], "by_k": {}} for q in qs}
    for k in ks:
        t0 = time.time()
        schedule = ["<search>"] * k + ["<answer>"]
        eps = rollout(model, tok, ret, qs, [0.0] * len(qs), max_calls=max(max(ks), 1),
                      greedy=True, forces=[schedule] * len(qs))
        for e in eps:
            queries = [m.strip() for t in e.transcript for m in SEARCH_RE.findall(t)]
            seen = {p["title"] for qu in queries for p in ret.search(qu)}
            gold = set(e.q["supporting_titles"])
            rows[e.q["id"]]["by_k"][str(k)] = {
                "answer": e.answer, "f1": f1_score(e.answer or "", e.q["answer"]),
                "em": exact_match(e.answer or "", e.q["answer"]), "n_search": e.n_search,
                "queries": queries, "gold_any": bool(gold & seen), "gold_both": gold <= seen}
        print(f"k={k} done in {time.time() - t0:.0f}s", flush=True)

    # the untrained agent left free to choose, on the same questions, at each price
    free = {}
    for lam in lams:
        eps = rollout(model, tok, ret, qs, [lam] * len(qs), max_calls=max(max(ks), 1), greedy=True)
        f1 = sum(f1_score(e.answer or "", e.q["answer"]) for e in eps) / len(eps)
        s = sum(e.n_search for e in eps) / len(eps)
        free[lam] = {"f1": f1, "searches": s, "utility": f1 - lam * s,
                     "hist": {k: sum(e.n_search == k for e in eps) / len(eps)
                              for k in range(max(ks) + 1)}}
        for e in eps:
            rows[e.q["id"]].setdefault("free", {})[str(lam)] = {
                "f1": f1_score(e.answer or "", e.q["answer"]), "n_search": e.n_search}
    print("free agent done", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".jsonl", "w", encoding="utf-8") as f:
        for r in rows.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    R = list(rows.values())
    summary = {"corpus": args.corpus, "n": len(R), "per_k": {}, "per_lam": {}}
    print(f"\n== per k (n={len(R)}, corpus={args.corpus}) ==")
    print("k  F1     95% CI           EM     no_ans  actual_searches  gold_any  gold_both")
    for k in ks:
        f1s = [r["by_k"][str(k)]["f1"] for r in R]
        lo, hi = bootstrap_ci(f1s)
        d = {"f1": sum(f1s) / len(R), "ci": [lo, hi],
             "em": sum(r["by_k"][str(k)]["em"] for r in R) / len(R),
             "no_answer": sum(r["by_k"][str(k)]["answer"] is None for r in R) / len(R),
             "actual_searches": sum(r["by_k"][str(k)]["n_search"] for r in R) / len(R),
             "gold_any": sum(r["by_k"][str(k)]["gold_any"] for r in R) / len(R),
             "gold_both": sum(r["by_k"][str(k)]["gold_both"] for r in R) / len(R)}
        summary["per_k"][k] = d
        print(f"{k}  {d['f1']:.3f}  [{lo:.3f}, {hi:.3f}]  {d['em']:.3f}  {d['no_answer']:.3f}   "
              f"{d['actual_searches']:.2f}             {d['gold_any']:.3f}     {d['gold_both']:.3f}")

    types = sorted({r["type"] for r in R})
    print("\n== utility = F1 - lambda*k ==")
    print("lambda  fixed(best k)   by_type          oracle   oracle-fixed  by_type-fixed  oracle k distribution")
    for lam in lams:
        u = {k: [r["by_k"][str(k)]["f1"] - lam * k for r in R] for k in ks}
        fixed_k = max(ks, key=lambda k: sum(u[k]))
        fixed = sum(u[fixed_k]) / len(R)
        by_type, type_k = 0.0, {}
        for t in types:
            idx = [i for i, r in enumerate(R) if r["type"] == t]
            tk = max(ks, key=lambda k: sum(u[k][i] for i in idx))
            type_k[t] = tk
            by_type += sum(u[tk][i] for i in idx)
        by_type /= len(R)
        # ties go to fewer searches
        best = [min(ks, key=lambda k: (-u[k][i], k)) for i in range(len(R))]
        oracle = sum(u[best[i]][i] for i in range(len(R))) / len(R)
        dist = {k: best.count(k) / len(R) for k in ks}
        summary["per_lam"][lam] = {"fixed": fixed, "fixed_k": fixed_k, "by_type": by_type,
                                   "type_k": type_k, "oracle": oracle, "oracle_k_dist": dist}
        print(f"{lam:<6}  {fixed:.3f} (k={fixed_k})   {by_type:.3f} {type_k}  {oracle:.3f}   "
              f"{oracle - fixed:+.3f}        {by_type - fixed:+.3f}         "
              + " ".join(f"{k}:{dist[k]:.2f}" for k in ks))

    print("\n== untrained agent left free vs best fixed k (same questions) ==")
    print("lambda  free F1  free searches  free utility  best-fixed utility  gap   search-count histogram")
    for lam in lams:
        fr, fx = free[lam], summary["per_lam"][lam]["fixed"]
        print(f"{lam:<6}  {fr['f1']:.3f}    {fr['searches']:.2f}           {fr['utility']:.3f}         "
              f"{fx:.3f}              {fx - fr['utility']:+.3f}  "
              + " ".join(f"{k}:{v:.2f}" for k, v in fr["hist"].items()))
    summary["free"] = free

    # how much of the oracle gain is per-question luck: questions where k does not matter
    all_zero = sum(all(r["by_k"][str(k)]["f1"] == 0 for k in ks) for r in R) / len(R)
    all_same = sum(len({r["by_k"][str(k)]["f1"] for k in ks}) == 1 for r in R) / len(R)
    nonmono = sum(any(r["by_k"][str(k)]["f1"] > r["by_k"][str(k + 1)]["f1"] for k in ks[:-1])
                  for r in R) / len(R)
    summary["k_irrelevant"] = {"all_zero": all_zero, "all_same": all_same, "non_monotone": nonmono}
    print(f"\nF1 = 0 for every k: {all_zero:.2f}; same F1 for every k: {all_same:.2f}; "
          f"F1 drops with an extra search somewhere: {nonmono:.2f}")

    print("\n== F1 by question type ==")
    for t in types:
        sub = [r for r in R if r["type"] == t]
        print(f"{t:<11} n={len(sub):<4} " + "  ".join(
            f"k={k}:{sum(r['by_k'][str(k)]['f1'] for r in sub) / len(sub):.3f}" for k in ks))
    with open(args.out + "_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
