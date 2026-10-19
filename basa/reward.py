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
                   format_penalty: float = 0.1) -> float:
    """F1 of the final answer minus the per-call search cost.

    `answer is None` means the episode ended without an <answer> tag
    (ran out of turns or broke the format)."""
    if answer is None:
        return -format_penalty - lam * n_search
    return f1_score(answer, gold) - lam * n_search
