"""BM25 search tool over the passage corpus."""
import bm25s
import Stemmer


class BM25Retriever:
    def __init__(self, corpus, k=3, max_chars=400):
        self.corpus, self.k, self.max_chars = corpus, k, max_chars
        self.stemmer = Stemmer.Stemmer("english")
        texts = [f"{p['title']}. {p['text']}" for p in corpus]
        tokens = bm25s.tokenize(texts, stopwords="en", stemmer=self.stemmer, show_progress=False)
        self.index = bm25s.BM25()
        self.index.index(tokens, show_progress=False)

    def search(self, query: str):
        q = bm25s.tokenize([query], stopwords="en", stemmer=self.stemmer, show_progress=False)
        if not q.ids or not q.ids[0]:
            return []
        ids, _ = self.index.retrieve(q, k=self.k, show_progress=False)
        return [self.corpus[i] for i in ids[0]]

    def format(self, query: str) -> str:
        hits = self.search(query)
        if not hits:
            return "No results."
        lines = []
        for i, p in enumerate(hits, 1):
            text = p["text"] if len(p["text"]) <= self.max_chars else p["text"][: self.max_chars] + "..."
            lines.append(f"[{i}] {p['title']}: {text}")
        return "\n".join(lines)
