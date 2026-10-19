"""Generate forced-k trajectories for the SFT cold start (see DEVLOG 6.6).

For every question we let the untrained model search exactly k times
(k = 0..3, its own queries, greedy) and then answer, and keep the token
sequence after the prompt together with its loss mask and score. basa/sft.py
then picks, per search price, which of the four trajectories to imitate.

  python scripts/make_sft_data.py --start 300 --n 2000 --corpus wiki
"""
import argparse
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from basa.data import build  # noqa: E402
from basa.retriever import BM25Retriever  # noqa: E402
from basa.reward import f1_score  # noqa: E402
from basa.rollout import SEARCH_RE, _prompt_ids, rollout  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--corpus", default="wiki", choices=["small", "wiki"])
    ap.add_argument("--start", type=int, default=300)  # skip the 300 diagnostic questions
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--max_calls", type=int, default=3)
    ap.add_argument("--out", default="cache/sft/traj_wiki.pt")
    args = ap.parse_args()

    train, _, corpus = build()
    qs = train[args.start:args.start + args.n]
    if args.corpus == "wiki":
        from basa.wiki import load_wiki_retriever
        ret = load_wiki_retriever()
    else:
        ret = BM25Retriever(corpus)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()

    out = {q["id"]: {"q": q, "by_k": {}} for q in qs}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for k in range(args.max_calls + 1):
        part = f"{args.out}.k{k}"
        if os.path.exists(part):  # resume: each k is saved as soon as it finishes
            for qid, v in torch.load(part).items():
                out[qid]["by_k"][k] = v
            print(f"k={k} loaded from {part}", flush=True)
            continue
        t0 = time.time()
        # contexts grow with every search; on a 16 GB card a batch of 24 at k=3
        # overflowed into shared system memory and slowed generation ~5x
        bs = {0: 24, 1: 24, 2: 12}.get(k, 8)
        eps = rollout(model, tok, ret, qs, [0.0] * len(qs), max_calls=args.max_calls, greedy=True,
                      forces=[["<search>"] * k + ["<answer>"]] * len(qs), batch_size=bs)
        for e in eps:
            n_prompt = len(_prompt_ids(tok, e.q["question"], 0.0, args.max_calls))
            queries = [m.strip() for t in e.transcript for m in SEARCH_RE.findall(t)]
            out[e.q["id"]]["by_k"][k] = {
                "ids": e.ids[n_prompt:], "mask": e.mask[n_prompt:], "answer": e.answer,
                "f1": f1_score(e.answer or "", e.q["answer"]), "n_search": e.n_search,
                "queries": queries,
                "repeat": len({q.lower() for q in queries}) < len(queries)}
        torch.save({qid: v["by_k"][k] for qid, v in out.items()}, part)
        print(f"k={k} done in {time.time() - t0:.0f}s", flush=True)

    torch.save(list(out.values()), args.out)
    for k in range(args.max_calls + 1):
        f1 = sum(r["by_k"][k]["f1"] for r in out.values()) / len(out)
        rep = sum(r["by_k"][k]["repeat"] for r in out.values()) / len(out)
        print(f"k={k} F1 {f1:.3f}  repeated-query rate {rep:.2f}")


if __name__ == "__main__":
    main()
