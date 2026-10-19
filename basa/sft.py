"""Price-conditioned SFT cold start (DEVLOG 6.6).

From the forced-k trajectories of scripts/make_sft_data.py we build, for each
question and a few search prices lambda, one imitation target: the trajectory
with the highest F1 - lambda * k. Ties go to the k closest to the best fixed
k for that price (from the diagnostic), so questions the model cannot answer
however much it searches teach the price-appropriate default rather than
"always give up". Trajectories that repeat a query are never targets.

Loss covers only tokens the model produced (search/answer openers, queries,
answers), never the prompt or the injected search results. When the target's
answer is wrong (F1 < 0.5) the answer text is masked as well, so we train the
decision to answer but not the wrong answer itself.

  python -m basa.sft --data cache/sft/traj_wiki.pt --out runs/sft_wiki
"""
import argparse
import json
import math
import os
import random
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .rollout import _prompt_ids

LAMS = [0.0, 0.05, 0.1, 0.2, 0.3]
FIXED_BEST = {0.0: 3, 0.05: 2, 0.1: 1, 0.2: 0, 0.3: 0}  # wiki-corpus diagnostic, DEVLOG 6.5


def choose_k(by_k, lam):
    ok = [k for k, t in by_k.items() if not t["repeat"]]
    u = {k: by_k[k]["f1"] - lam * k for k in ok}
    top = max(u.values())
    best = [k for k in ok if u[k] >= top - 1e-9]
    return min(best, key=lambda k: (abs(k - FIXED_BEST[lam]), k))


def _find_last(seq, sub):
    for i in range(len(seq) - len(sub), -1, -1):
        if seq[i:i + len(sub)] == sub:
            return i
    return -1


def build_examples(records, tok, max_calls, lams_per_q, seed):
    rng = random.Random(seed)
    opener = tok("<answer>", add_special_tokens=False).input_ids
    examples, stats = [], {str(l): {str(k): 0 for k in range(max_calls + 1)} for l in LAMS}
    n_masked = 0
    for r in records:
        by_k = {int(k): v for k, v in r["by_k"].items()}
        for lam in rng.sample(LAMS, lams_per_q):
            k = choose_k(by_k, lam)
            t = by_k[k]
            prompt = _prompt_ids(tok, r["q"]["question"], lam, max_calls)
            labels = [x if m else -100 for x, m in zip(t["ids"], t["mask"])]
            if t["f1"] < 0.5:
                j = _find_last(t["ids"], opener)
                if j >= 0:
                    for p in range(j + len(opener), len(labels)):
                        labels[p] = -100
                    n_masked += 1
            examples.append({"ids": prompt + t["ids"], "labels": [-100] * len(prompt) + labels,
                             "lam": lam, "k": k})
            stats[str(lam)][str(k)] += 1
    return examples, stats, n_masked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--data", default="cache/sft/traj_wiki.pt")
    ap.add_argument("--out", default="runs/sft_wiki")
    ap.add_argument("--max_calls", type=int, default=3)
    ap.add_argument("--lams_per_q", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--micro", type=int, default=4)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    records = torch.load(args.data)
    examples, stats, n_masked = build_examples(records, tok, args.max_calls, args.lams_per_q, args.seed)
    print(f"{len(examples)} examples, answer text masked in {n_masked}")
    print("target k per lambda:", json.dumps(stats))
    with open(os.path.join(args.out, "data_stats.json"), "w") as f:
        json.dump({"n": len(examples), "answer_masked": n_masked, "k_per_lambda": stats}, f, indent=1)

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda()
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps = math.ceil(len(examples) / args.batch) * args.epochs
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (steps - s) / max(1, steps - warm)))

    log = open(os.path.join(args.out, "train_log.jsonl"), "w")
    step, t0 = 0, time.time()
    for ep in range(args.epochs):
        random.shuffle(examples)
        for b in range(0, len(examples), args.batch):
            batch = examples[b:b + args.batch]
            n_lab = sum(sum(x != -100 for x in e["labels"][1:]) for e in batch)
            opt.zero_grad(set_to_none=True)
            total = 0.0
            for m in range(0, len(batch), args.micro):
                mb = batch[m:m + args.micro]
                w = max(len(e["ids"]) for e in mb)
                ids = torch.full((len(mb), w), tok.pad_token_id)
                lab = torch.full((len(mb), w), -100)
                att = torch.zeros((len(mb), w), dtype=torch.long)
                for i, e in enumerate(mb):
                    ids[i, :len(e["ids"])] = torch.tensor(e["ids"])
                    lab[i, :len(e["labels"])] = torch.tensor(e["labels"])
                    att[i, :len(e["ids"])] = 1
                ids, lab, att = ids.cuda(), lab.cuda(), att.cuda()
                hs = model.get_base_model().model(input_ids=ids, attention_mask=att).last_hidden_state
                loss = 0.0
                for s in range(0, w - 1, 512):  # chunked LM head, as in train_grpo.seq_logprobs
                    h = hs[:, s:min(s + 512, w - 1)]
                    logits = model.get_base_model().lm_head(h).float()
                    tgt = lab[:, s + 1:s + 1 + h.shape[1]]
                    loss = loss + torch.nn.functional.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1),
                        ignore_index=-100, reduction="sum")
                loss = loss / n_lab
                loss.backward()
                total += loss.item()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1
            log.write(json.dumps({"step": step, "loss": total, "lr": sched.get_last_lr()[0]}) + "\n")
            log.flush()
            if step % 20 == 0 or step == steps:
                print(f"step {step}/{steps} loss {total:.4f} {time.time() - t0:.0f}s", flush=True)
    model.save_pretrained(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
