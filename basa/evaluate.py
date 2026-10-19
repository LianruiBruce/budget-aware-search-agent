"""Evaluate baselines and trained adapters across search costs.

Methods
  no_search   the agent may not search (max_calls=0): closed-book answer
  rag1        one retrieval with the question as query, results given up front
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
from .rollout import ANSWER_RE, _generate, rollout


def summarize(eps, lam):
    n = len(eps)
    f1 = sum(f1_score(e.answer or "", e.q["answer"]) for e in eps) / n
    em = sum(exact_match(e.answer or "", e.q["answer"]) for e in eps) / n
    s = sum(e.n_search for e in eps) / n
    return {"lam": lam, "em": em, "f1": f1, "searches": s, "utility": f1 - lam * s,
            "no_answer": sum(e.answer is None for e in eps) / n}


def rag1(model, tok, ret, dev, batch=32):
    out = []
    for b in range(0, len(dev), batch):
        chunk = dev[b:b + batch]
        seqs = []
        for q in chunk:
            msgs = [{"role": "system", "content": "Answer the question using the passages. "
                     "Write <answer>short answer</answer>."},
                    {"role": "user", "content": f"Passages:\n{ret.format(q['question'])}\n\n"
                     f"Question: {q['question']}"}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
            seqs.append(tok(text, add_special_tokens=False).input_ids)
        for q, g in zip(chunk, _generate(model, tok, seqs, 64, 1.0, True)):
            m = ANSWER_RE.search(tok.decode(g, skip_special_tokens=True))
            out.append(type("E", (), {"q": q, "answer": m.group(1).strip() if m else None,
                                      "n_search": 1})())
    return out


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
            emit(method, summarize(rag1(model, tok, ret, dev), 0.0))
        else:  # react (base) or trained (adapter): the agent prompt at each cost
            for lam in lams:
                eps = rollout(model, tok, ret, dev, [lam] * len(dev), max_calls=args.max_calls,
                              greedy=True)
                emit(method, summarize(eps, lam))


if __name__ == "__main__":
    main()
