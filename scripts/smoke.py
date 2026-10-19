"""Quick end-to-end check: retriever recall, then a few greedy rollouts."""
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, ".")
from basa.data import build
from basa.retriever import BM25Retriever
from basa.reward import f1_score
from basa.rollout import rollout

train, dev, corpus = build()
t = time.time()
ret = BM25Retriever(corpus)
print(f"index built in {time.time() - t:.1f}s")

# recall@3 of at least one supporting passage when querying with the raw question
hit = sum(any(p["title"] in q["supporting_titles"] for p in ret.search(q["question"])) for q in dev)
both = sum(all(t in {p["title"] for p in ret.search(q["question"])} for t in q["supporting_titles"]) for q in dev)
print(f"question-as-query: any-gold@3 {hit / len(dev):.3f}, both-gold@3 {both / len(dev):.3f}")

name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-1.7B"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).cuda().eval()
qs = dev[:8]
t = time.time()
eps = rollout(model, tok, ret, qs, [0.1] * len(qs), greedy=True)
print(f"8 greedy rollouts in {time.time() - t:.1f}s")
for e in eps:
    print("=" * 80)
    print("Q:", e.q["question"], "| gold:", e.q["answer"])
    print("\n".join(x[:300] for x in e.transcript))
    print(f"-> answer={e.answer!r} searches={e.n_search} f1={f1_score(e.answer or '', e.q['answer']):.2f}"
          f" gen_tokens={sum(e.mask)} total={len(e.ids)}")
