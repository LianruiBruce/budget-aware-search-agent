"""Answer normalization and scoring (HotpotQA / SQuAD-style EM and F1)."""
import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred), normalize_answer(gold)
    # yes/no/noanswer must match exactly, as in the official HotpotQA script
    if p in {"yes", "no", "noanswer"} or g in {"yes", "no", "noanswer"}:
        return float(p == g)
    pt, gt = p.split(), g.split()
    common = Counter(pt) & Counter(gt)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(pt), same / len(gt)
    return 2 * precision * recall / (precision + recall)


def episode_reward(answer, gold: str, n_search: int, lam: float,
                   format_penalty: float = 0.1, mode: str = "additive") -> float:
    """Training reward for one episode.

    additive  F1 - lambda * searches: the true utility; failed searches cost too
    scaled    F1 * (1 - lambda * searches): the cost only bites when the answer
              is right (OTC-PO-style), so failed searches are not punished

    `answer is None` means the episode ended without an <answer> tag
    (ran out of turns or broke the format)."""
    if mode == "scaled":
        if answer is None:
            return -format_penalty
        return f1_score(answer, gold) * (1 - lam * n_search)
    if answer is None:
        return -format_penalty - lam * n_search
    return f1_score(answer, gold) - lam * n_search
