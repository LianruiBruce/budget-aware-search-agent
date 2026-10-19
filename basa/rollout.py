"""Multi-turn agent rollouts with tool calls.

Every episode is kept as one token sequence plus a mask: 1 on tokens the
policy generated (trained on), 0 on the prompt and on tool results we
injected (never trained on).
"""
import re
from dataclasses import dataclass, field

import torch

from .prompts import build_messages

STOPS = ["</search>", "</answer>"]
SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.S)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S)


@dataclass
class Episode:
    q: dict
    lam: float
    ids: list
    mask: list
    n_search: int = 0
    answer: str = None
    done: bool = False
    transcript: list = field(default_factory=list)


def _prompt_ids(tok, question, lam, max_calls):
    msgs = build_messages(question, lam, max_calls)
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    return tok(text, add_special_tokens=False).input_ids


@torch.no_grad()
def _generate(model, tok, seqs, max_new_tokens, temperature, greedy):
    pad = tok.pad_token_id
    width = max(len(s) for s in seqs)
    inp = torch.full((len(seqs), width), pad, dtype=torch.long)
    att = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):  # left padding
        inp[i, width - len(s):] = torch.tensor(s)
        att[i, width - len(s):] = 1
    inp, att = inp.to(model.device), att.to(model.device)
    kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=pad,
                  stop_strings=STOPS, tokenizer=tok)
    if greedy:
        kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    else:
        kwargs.update(do_sample=True, temperature=temperature, top_p=1.0, top_k=0)
    out = model.generate(input_ids=inp, attention_mask=att, **kwargs)
    gens = []
    for row in out[:, width:].tolist():
        while row and row[-1] == pad:
            row.pop()
        gens.append(row)
    return gens


def rollout(model, tok, retriever, questions, lams, max_calls=4, max_new_tokens=200,
            temperature=1.0, greedy=False, max_ctx=3072, batch_size=24):
    eps = []
    for q, lam in zip(questions, lams):
        ids = _prompt_ids(tok, q["question"], lam, max_calls)
        eps.append(Episode(q=q, lam=lam, ids=ids, mask=[0] * len(ids)))

    eos = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    for _ in range(max_calls + 1):
        active = [e for e in eps if not e.done]
        if not active:
            break
        for b in range(0, len(active), batch_size):
            chunk = active[b:b + batch_size]
            gens = _generate(model, tok, [e.ids for e in chunk], max_new_tokens, temperature, greedy)
            for e, g in zip(chunk, gens):
                _step(e, g, tok, retriever, max_calls, eos)
                if len(e.ids) > max_ctx and not e.done:
                    e.done = True
    for e in eps:
        e.done = True
    return eps


def _inject(e, tok, text):
    t = tok(text, add_special_tokens=False).input_ids
    e.ids += t
    e.mask += [0] * len(t)


def _step(e, gen, tok, retriever, max_calls, eos):
    e.ids += gen
    e.mask += [1] * len(gen)
    text = tok.decode(gen, skip_special_tokens=True)
    e.transcript.append(text)

    m = ANSWER_RE.search(text)
    if m:
        e.answer, e.done = m.group(1).strip(), True
        return
    m = SEARCH_RE.search(text)
    if m:
        if e.n_search >= max_calls:
            _inject(e, tok, "\n<result>\nSearch budget exhausted. Answer now.\n</result>\n")
            e.transcript.append("[budget exhausted]")
            return
        e.n_search += 1
        result = retriever.format(m.group(1).strip())
        _inject(e, tok, f"\n<result>\n{result}\n</result>\n")
        e.transcript.append(f"<result>{result}</result>")
        return
    # neither tag: the model stopped (EOS) or ran out of tokens without acting
    e.done = True
