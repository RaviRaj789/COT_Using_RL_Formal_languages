import argparse
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import trange

from .evaluate import evaluate_buckets
from .model import CausalTransformerPolicy
from .rl import collect_rollout, grpo_update, ppo_update
from .tasks import TRAIN_MAX_LEN, TRAIN_MIN_LEN, build_task
from .tokenizer import Tokenizer
from .utils import append_jsonl, device_auto, load_config, save_json, seed_everything


def build_model(cfg: dict, tokenizer: Tokenizer) -> CausalTransformerPolicy:
    mcfg = cfg.get("model", {})
    return CausalTransformerPolicy(
        vocab_size=len(tokenizer.id_to_token),
        d_model=mcfg.get("d_model", 128),
        n_heads=mcfg.get("n_heads", 4),
        n_layers=mcfg.get("n_layers", 4),
        d_ff=mcfg.get("d_ff", 512),
        dropout=mcfg.get("dropout", 0.0),
        max_seq_len=mcfg.get("max_seq_len", 2048),
        pad_id=tokenizer.pad_id,
    )


def sample_examples(task, rng: random.Random, batch_size: int, min_len: int, max_len: int):
    return [task.sample(rng.randint(min_len, max_len), rng) for _ in range(batch_size)]


def log_train_metrics(
    run_dir: Path,
    history: list[dict],
    step: int,
    alg: str,
    metrics: dict[str, float],
    started_at: float,
) -> None:
    record = {
        "step": step,
        "algorithm": alg,
        "elapsed_sec": time.time() - started_at,
        "train": metrics,
    }
    history.append(record)
    append_jsonl(record, run_dir / "train_metrics.jsonl")
    save_json(history, run_dir / "train_metrics.json")
    print(f"[train] step={step} metrics={metrics}", flush=True)


def sft_step(model, optimizer, tokenizer, examples, device):
    rows = []
    masks = []
    for ex in examples:
        prompt = tokenizer.encode(ex.prompt_tokens, add_bos=True)
        target = tokenizer.encode(ex.target_tokens, add_eos=True)
        rows.append(prompt + target)
        masks.append([False] * len(prompt) + [True] * len(target))
    batch, _ = tokenizer.pad_batch(rows)
    action_mask, _ = tokenizer.pad_batch([[int(x) for x in m] for m in masks])
    batch = batch.to(device)
    action_mask = action_mask.bool().to(device)
    logits, _ = model(batch[:, :-1])
    targets = batch[:, 1:]
    loss = F.cross_entropy(logits[action_mask[:, 1:]], targets[action_mask[:, 1:]])
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    preds = logits[action_mask[:, 1:]].argmax(dim=-1)
    acc = (preds == targets[action_mask[:, 1:]]).float().mean()
    return {"loss": float(loss.item()), "token_acc": float(acc.item())}


def train(cfg: dict) -> Path:
    rng = seed_everything(cfg.get("seed", 0))
    device = device_auto()
    tokenizer = Tokenizer.default()
    task = build_task(cfg["task"]["name"])
    model = build_model(cfg, tokenizer).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"].get("lr", 3e-4))
    run_dir = Path("runs") / cfg.get("run_name", f"{task.name}_{cfg['algorithm']['name']}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "train_metrics.jsonl").write_text("", encoding="utf-8")
    save_json([], run_dir / "train_metrics.json")
    steps = cfg["train"].get("steps", 1000)
    batch_size = cfg["train"].get("batch_size", 64)
    min_len = cfg["train"].get("train_min_len", TRAIN_MIN_LEN)
    max_len = cfg["train"].get("train_max_len", TRAIN_MAX_LEN)
    if min_len < 1 or max_len > 40:
        raise ValueError("Training lengths must stay within 1..40.")
    alg = cfg["algorithm"]["name"]
    gen = cfg.get("generation", {})
    max_new = gen.get("max_new_tokens", 520)
    temp = gen.get("temperature", 1.0)
    progress = trange(1, steps + 1, desc=alg)
    last_metrics = {}
    train_history = []
    started_at = time.time()
    log_every = cfg["train"].get("log_every", 50)
    for step in progress:
        model.train()
        if alg == "sft":
            examples = sample_examples(task, rng, batch_size, min_len, max_len)
            last_metrics = sft_step(model, optimizer, tokenizer, examples, device)
        elif alg.startswith("ppo_"):
            acfg = cfg["algorithm"]
            rollouts = [
                collect_rollout(
                    model,
                    tokenizer,
                    task,
                    ex,
                    alg,
                    max_new,
                    temp,
                    device,
                    acfg.get("process_weight", 1.0),
                    acfg.get("terminal_weight", 1.0),
                )
                for ex in sample_examples(task, rng, acfg.get("rollout_batch_size", batch_size), min_len, max_len)
            ]
            last_metrics = ppo_update(
                model,
                optimizer,
                tokenizer,
                rollouts,
                acfg.get("clip_eps", 0.2),
                acfg.get("value_coef", 0.5),
                acfg.get("entropy_coef", 0.01),
                acfg.get("ppo_epochs", 4),
                device,
            )
        elif alg.startswith("grpo_"):
            acfg = cfg["algorithm"]
            groups = []
            for ex in sample_examples(task, rng, acfg.get("groups_per_step", batch_size), min_len, max_len):
                groups.append(
                    [
                        collect_rollout(
                            model,
                            tokenizer,
                            task,
                            ex,
                            alg,
                            max_new,
                            temp,
                            device,
                            acfg.get("process_weight", 1.0),
                            acfg.get("terminal_weight", 1.0),
                        )
                        for _ in range(acfg.get("samples_per_prompt", 4))
                    ]
                )
            last_metrics = grpo_update(
                model, optimizer, tokenizer, groups, acfg.get("clip_eps", 0.2), acfg.get("entropy_coef", 0.01), device
            )
        else:
            raise ValueError(f"Unknown algorithm: {alg}")
        progress.set_postfix(last_metrics)
        if log_every and step % log_every == 0:
            log_train_metrics(run_dir, train_history, step, alg, last_metrics, started_at)
        if step % cfg["train"].get("eval_every", 200) == 0:
            metrics = evaluate_buckets(model, tokenizer, task, 1, max_new, device)
            save_json({"step": step, "train": last_metrics, "eval": metrics}, run_dir / f"metrics_step{step}.json")
    ckpt = {
        "config": cfg,
        "model": model.state_dict(),
        "vocab": tokenizer.id_to_token,
    }
    ckpt_path = run_dir / "checkpoint.pt"
    torch.save(ckpt, ckpt_path)
    if log_every and steps % log_every != 0:
        log_train_metrics(run_dir, train_history, steps, alg, last_metrics, started_at)
    save_json({"final_train_metrics": last_metrics, "train_log_every": log_every}, run_dir / "summary.json")
    return ckpt_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--log-every", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.steps is not None:
        cfg["train"]["steps"] = args.steps
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.log_every is not None:
        cfg["train"]["log_every"] = args.log_every
    path = train(cfg)
    print(path)


if __name__ == "__main__":
    main()
