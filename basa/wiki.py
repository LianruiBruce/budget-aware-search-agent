"""The full HotpotQA Wikipedia corpus: first paragraphs (abstracts) of every
English Wikipedia article in the 2017-10-01 dump, about 5 million passages.

HotpotQA questions were written from these abstracts, so the gold paragraphs
are in here too, but now among millions of others instead of the 75k context
paragraphs of the small corpus, so finding them takes real queries.

  python -m basa.wiki   # stream the archive into abstracts.jsonl, build the BM25 index
"""
import bz2
import json
import os
import tarfile
import time

import bm25s
import Stemmer

from .retriever import BM25Retriever

DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache", "wiki")
ARCHIVE = os.path.join(DIR, "enwiki-abstracts.tar.bz2")
CORPUS = os.path.join(DIR, "abstracts.jsonl")
INDEX = os.path.join(DIR, "bm25_index")


def prepare_corpus():
    """Stream the tar.bz2 (one bz2 file of JSON lines per member) into a plain
    title/text JSONL. Empty abstracts are dropped."""
    n, t0 = 0, time.time()
    with tarfile.open(ARCHIVE, "r|bz2") as tar, open(CORPUS, "w", encoding="utf-8") as out:
        for m in tar:
            if not (m.isfile() and m.name.endswith(".bz2")):
                continue
            for line in bz2.decompress(tar.extractfile(m).read()).decode("utf-8").splitlines():
                d = json.loads(line)
                text = "".join(d["text"]).strip()
                if text:
                    out.write(json.dumps({"title": d["title"], "text": text}, ensure_ascii=False) + "\n")
                    n += 1
            if n and n % 500000 < 400:
                print(f"{n} passages, {time.time() - t0:.0f}s", flush=True)
    print(f"wrote {n} passages to {CORPUS} in {time.time() - t0:.0f}s")


def load_corpus():
    with open(CORPUS, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_index(corpus):
    stemmer = Stemmer.Stemmer("english")
    t0 = time.time()
    tokens = bm25s.tokenize([f"{p['title']}. {p['text']}" for p in corpus], stopwords="en",
                            stemmer=stemmer, show_progress=True)
    index = bm25s.BM25()
    index.index(tokens, show_progress=True)
    index.save(INDEX)
    print(f"indexed {len(corpus)} passages in {time.time() - t0:.0f}s")


class PrebuiltBM25Retriever(BM25Retriever):
    """BM25Retriever over an index that was built and saved ahead of time."""

    def __init__(self, corpus, index, k=3, max_chars=400, n_threads=16):
        self.corpus, self.k, self.max_chars, self.n_threads = corpus, k, max_chars, n_threads
        self.stemmer = Stemmer.Stemmer("english")
        self.index = index


def load_wiki_retriever(k=3):
    return PrebuiltBM25Retriever(load_corpus(), bm25s.BM25.load(INDEX, mmap=False), k=k)


if __name__ == "__main__":
    if not os.path.exists(CORPUS):
        prepare_corpus()
    if not os.path.exists(INDEX):
        build_index(load_corpus())
