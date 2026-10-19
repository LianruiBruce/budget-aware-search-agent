"""HotpotQA loading and the shared passage corpus the agent searches over.

The corpus is the union of every context paragraph (gold + distractor) of the
questions we use, deduplicated by title. Each question therefore has its two
supporting paragraphs somewhere in a pool of tens of thousands of passages, and
the agent has to find them with its own queries.
"""
import json
import os
import random

from datasets import load_dataset

CACHE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache")


def _rows(split, n, seed):
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    return [ds[i] for i in idx[:n]]


def build(n_train=8000, n_dev=500, seed=0):
    """Returns (train, dev, corpus) and caches them as JSON."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"hotpot_{n_train}_{n_dev}_{seed}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return d["train"], d["dev"], d["corpus"]

    train_rows = _rows("train", n_train, seed)
    dev_rows = _rows("validation", n_dev, seed)
    corpus, seen = [], set()

    def add_context(row):
        for title, sents in zip(row["context"]["title"], row["context"]["sentences"]):
            if title in seen:
                continue
            seen.add(title)
            corpus.append({"title": title, "text": "".join(sents).strip()})

    def to_q(row):
        return {"id": row["id"], "question": row["question"], "answer": row["answer"],
                "type": row["type"], "level": row["level"],
                "supporting_titles": sorted(set(row["supporting_facts"]["title"]))}

    for row in train_rows + dev_rows:
        add_context(row)
    train = [to_q(r) for r in train_rows]
    dev = [to_q(r) for r in dev_rows]
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"train": train, "dev": dev, "corpus": corpus}, f, ensure_ascii=False)
    return train, dev, corpus


if __name__ == "__main__":
    tr, dv, cp = build()
    print(f"train={len(tr)} dev={len(dv)} corpus={len(cp)} passages")
    print(tr[0])
