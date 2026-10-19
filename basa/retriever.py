"""BM25 search tool over the passage corpus."""
import bm25s
import Stemmer


class BM25Retriever:
    def __init__(self, corpus, k=3, max_chars=400, n_threads=16):
        self.corpus, self.k, self.max_chars, self.n_threads = corpus, k, max_chars, n_threads
        self.stemmer = Stemmer.Stemmer("english")
        texts = [f"{p['title']}. {p['text']}" for p in corpus]
        tokens = bm25s.tokenize(texts, stopwords="en", stemmer=self.stemmer, show_progress=False)
        self.index = bm25s.BM25()
        self.index.index(tokens, show_progress=False)

    @property
    def _cache(self):
        # created lazily so subclasses that skip __init__ get one too
        if not hasattr(self, "_c"):
            self._c = {}
        return self._c

    def prefetch(self, queries):
        """Retrieve many queries in one multi-threaded call and cache the hits.
        On the 5M-passage Wikipedia index one query takes ~0.3 s single-threaded,
        so rollouts batch every search of a generation step through here."""
        todo = list(dict.fromkeys(q for q in queries if q not in self._cache))
        if not todo:
            return
        toks = bm25s.tokenize(todo, stopwords="en", stemmer=self.stemmer, show_progress=False)
        keep = [i for i, ids in enumerate(toks.ids) if ids]
        for i, q in enumerate(todo):
            if i not in set(keep):
                self._cache[q] = []
        if not keep:
            return
        sub = bm25s.tokenization.Tokenized(ids=[toks.ids[i] for i in keep], vocab=toks.vocab)
        ids, _ = self.index.retrieve(sub, k=self.k, show_progress=False, n_threads=self.n_threads)
        for row, i in zip(ids, keep):
            self._cache[todo[i]] = [self.corpus[j] for j in row]

    def search(self, query: str):
        if query not in self._cache:
            self.prefetch([query])
        return self._cache[query]

    def format(self, query: str) -> str:
        hits = self.search(query)
        if not hits:
            return "No results."
        lines = []
        for i, p in enumerate(hits, 1):
            text = p["text"] if len(p["text"]) <= self.max_chars else p["text"][: self.max_chars] + "..."
            lines.append(f"[{i}] {p['title']}: {text}")
        return "\n".join(lines)
