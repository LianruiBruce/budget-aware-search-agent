SYSTEM = (
    "You answer multi-hop questions by searching a Wikipedia passage index.\n"
    "Search cost: {lam:.2f} points per search. A fully correct answer earns 1 point, "
    "so only search when it is worth the cost.\n"
    "To search, write <search>your query</search> and stop; the results come back inside "
    "<result></result>. You may search at most {max_calls} times.\n"
    "When you are ready, write <answer>short answer</answer>. "
    "Keep answers short (an entity, a date, yes or no)."
)


def build_messages(question: str, lam: float, max_calls: int):
    return [
        {"role": "system", "content": SYSTEM.format(lam=lam, max_calls=max_calls)},
        {"role": "user", "content": f"Question: {question}"},
    ]
