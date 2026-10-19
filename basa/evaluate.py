"""Evaluate baselines and trained adapters across search costs.

Methods
  no_search   the agent may not search (max_calls=0): closed-book answer
  rag1        one search with the question as query is done for the agent up front;
              it may not search again
  react       the untrained model with the agent prompt (prompted ReAct-style tool use)
  trained     the GRPO adapter with the same prompt

For each search cost lambda we report EM, F1, mean searches and the utility
F1 - lambda * searches that the agent is trained to maximize.
"""
import argparse
import json
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import build
from .retriever import BM25Retriever
from .reward import exact_match, f1_score
from .rollout import rollout


def summarize(eps, lam):
    n = len(eps)
    f1 = sum(f1_score(e.answer or "", e.q["answer"]) for e in eps) / n
    em = sum(exact_match(e.answer or "", e.q["answer"]) for e in eps) / n
    s = sum(e.n_search for e in eps) / n
    return {"lam": lam, "em": em, "f1": f1, "searches": s, "utility": f1 - lam * s,
            "no_answer": sum(e.answer is None for e in eps) / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--methods", default="no_search,rag1,react")
    ap.add_argument("--lams", default="0,0.05,0.1,0.2,0.3")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--max_calls", type=int, default=4)
    ap.add_argument("--out", default="results/eval.jsonl")
    args = ap.parse_args()

    _, dev, corpus = build()
    dev = dev[:args.n]
    ret = BM25Retriever(corpus)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter).eval()
    lams = [float(x) for x in args.lams.split(",")]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    f = open(args.out, "a", encoding="utf-8")

    def emit(method, rec):
        rec = {"method": method, "adapter": args.adapter, **rec}
        print(json.dumps(rec), flush=True)
        f.write(json.dumps(rec) + "\n")
        f.flush()

    for method in args.methods.split(","):
        if method == "no_search":
            eps = rollout(model, tok, ret, dev, [0.0] * len(dev), max_calls=0, greedy=True)
            emit(method, summarize(eps, 0.0))
        elif method == "rag1":
            eps = rollout(model, tok, ret, dev, [0.0] * len(dev), max_calls=1, greedy=True,
                          prefill_search=True)
            emit(method, summarize(eps, 0.0))
        else:  # react (base) or trained (adapter): the agent prompt at each cost
            for lam in lams:
                eps = rollout(model, tok, ret, dev, [lam] * len(dev), max_calls=args.max_calls,
                              greedy=True)
                emit(method, summarize(eps, lam))


if __name__ == "__main__":
    main()
