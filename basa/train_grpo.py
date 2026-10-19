"""Budget-conditioned GRPO for a multi-turn search agent.

Each question is paired with a search cost lambda that is written into the
prompt. The group of G rollouts for a question shares that lambda, so the
group-normalized advantage compares answers that faced the same price.
Reward = F1(answer) - lambda * (#searches); a missing answer costs 0.1.
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

from .data import build
from .retriever import BM25Retriever
from .reward import episode_reward, f1_score
from .rollout import rollout


def seq_logprobs(model, ids, attn, chunk=512):
    """Log-probs of ids[:, 1:]. The LM head runs over 512-position chunks so the
    full [T, 151k-vocab] float logits never sit in memory at once."""
    base = model.get_base_model()
    hs = base.model(input_ids=ids, attention_mask=attn).last_hidden_state
    T = ids.shape[1]
    out = []
    for s in range(0, T - 1, chunk):
        h = hs[:, s:min(s + chunk, T - 1)]
        logits = base.lm_head(h).float()
        tgt = ids[:, s + 1:s + 1 + h.shape[1]]
        out.append(torch.log_softmax(logits, -1).gather(-1, tgt[..., None]).squeeze(-1))
    return torch.cat(out, 1)


def policy_loss(model, eps, advs, pad_id, device, micro=1):
    """Sum of -A * log pi over generated tokens, normalized by the total number
    of generated tokens in the batch (a constant, so long and short episodes
    are weighted per token, not per episode)."""
    n_tok = sum(sum(e.mask) for e in eps)
    total = 0.0
    for b in range(0, len(eps), micro):
        chunk, a = eps[b:b + micro], advs[b:b + micro]
        width = max(len(e.ids) for e in chunk)
        ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
        attn = torch.zeros_like(ids)
        mask = torch.zeros((len(chunk), width), dtype=torch.float)
        for i, e in enumerate(chunk):  # right padding for training
            ids[i, :len(e.ids)] = torch.tensor(e.ids)
            attn[i, :len(e.ids)] = 1
            mask[i, :len(e.mask)] = torch.tensor(e.mask, dtype=torch.float)
        ids, attn, mask = ids.to(device), attn.to(device), mask.to(device)
        lp = seq_logprobs(model, ids, attn)
        adv = torch.tensor(a, dtype=torch.float, device=device)[:, None]
        loss = -(adv * lp * mask[:, 1:]).sum() / n_tok
        loss.backward()
        total += loss.item()
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--q_per_step", type=int, default=4)
    ap.add_argument("--group", type=int, default=6)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lams", default="0,0.05,0.1,0.2,0.3")
    ap.add_argument("--max_calls", type=int, default=4)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--out", default="runs/basa")
    ap.add_argument("--save_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    lams = [float(x) for x in args.lams.split(",")]

    train, _, corpus = build()
    retriever = BM25Retriever(corpus)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").cuda()
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    model.enable_input_require_grads()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr,
                            weight_decay=0.0)

    log = open(os.path.join(args.out, "train_log.jsonl"), "a", encoding="utf-8")
    for step in range(1, args.steps + 1):
        t0 = time.time()
        qs = random.sample(train, args.q_per_step)
        q_lams = [random.choice(lams) for _ in qs]
        batch_q = [q for q in qs for _ in range(args.group)]
        batch_l = [l for l in q_lams for _ in range(args.group)]

        model.eval()
        model.gradient_checkpointing_disable()
        model.config.use_cache = True
        eps = rollout(model, tok, retriever, batch_q, batch_l, max_calls=args.max_calls)
        rewards = [episode_reward(e.answer, e.q["answer"], e.n_search, e.lam) for e in eps]

        # group-normalized advantages; groups with no reward spread carry no signal
        keep_eps, advs = [], []
        for g in range(args.q_per_step):
            r = rewards[g * args.group:(g + 1) * args.group]
            mu = sum(r) / len(r)
            sd = math.sqrt(sum((x - mu) ** 2 for x in r) / len(r))
            if sd < 1e-6:
                continue
            for e, x in zip(eps[g * args.group:(g + 1) * args.group], r):
                keep_eps.append(e)
                advs.append((x - mu) / (sd + 1e-4))

        loss = 0.0
        if keep_eps:
            model.train()
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.config.use_cache = False
            opt.zero_grad(set_to_none=True)
            loss = policy_loss(model, keep_eps, advs, tok.pad_token_id, model.device)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()

        f1s = [f1_score(e.answer or "", e.q["answer"]) for e in eps]
        rec = {
            "step": step, "loss": loss, "reward": sum(rewards) / len(rewards),
            "f1": sum(f1s) / len(f1s),
            "searches": sum(e.n_search for e in eps) / len(eps),
            "no_answer": sum(e.answer is None for e in eps) / len(eps),
            "kept_groups": len(keep_eps) // args.group,
            "gen_tokens": sum(sum(e.mask) for e in eps), "sec": round(time.time() - t0, 1),
            "by_lam": {str(l): {
                "searches": sum(e.n_search for e in eps if e.lam == l) / max(1, sum(e.lam == l for e in eps)),
                "f1": sum(f for f, e in zip(f1s, eps) if e.lam == l) / max(1, sum(e.lam == l for e in eps))}
                for l in sorted(set(batch_l))},
        }
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"step {step} reward {rec['reward']:.3f} f1 {rec['f1']:.3f} "
              f"searches {rec['searches']:.2f} no_ans {rec['no_answer']:.2f} "
              f"kept {rec['kept_groups']} {rec['sec']}s", flush=True)
        if step % args.save_every == 0 or step == args.steps:
            model.save_pretrained(os.path.join(args.out, f"step{step}"))
            with open(os.path.join(args.out, f"step{step}", "sample.txt"), "w", encoding="utf-8") as f:
                for e in eps[:args.group]:
                    f.write(f"Q: {e.q['question']} | gold: {e.q['answer']} | lam={e.lam}\n")
                    f.write("\n".join(e.transcript) + "\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()
