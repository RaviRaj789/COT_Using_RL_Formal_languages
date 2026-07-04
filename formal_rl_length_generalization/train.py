import argparse
import copy
import random
import time
from pathlib import Path
from typing import Optional

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


def sample_examples(
    task,
    rng: random.Random,
    batch_size: int,
    min_len: int,
    max_len: int,
):
    return [task.sample(rng.randint(min_len, max_len), rng) for _ in range(batch_size)]


def resolve_run_dir(cfg: dict, task_name: str, alg: str) -> Path:
    base = Path("runs") / cfg.get("run_name", f"{task_name}_{alg}")

    overwrite = bool(cfg.get("train", {}).get("overwrite_run", False))

    if overwrite or not base.exists():
        base.mkdir(parents=True, exist_ok=True)
        return base

    for i in range(1, 1000):
        candidate = base.with_name(f"{base.name}_{i:03d}")

        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            print(f"Run directory already exists: {base}", flush=True)
            print(f"Writing this run to: {candidate}", flush=True)
            return candidate

    raise RuntimeError(f"Could not create a non-overwriting run directory for base: {base}")


def resolve_reward_kwargs(acfg: dict, stage: dict) -> dict:
    """Resolve reward-related knobs from the curriculum stage, falling back to algorithm config.

    reward_mode="sequence" reproduces the original single-scalar-per-token reward exactly;
    reward_mode="dense" turns on per-step partial-credit rewards from dense_reward.py.
    """
    return {
        "process_weight": float(stage.get("process_weight", acfg.get("process_weight", 1.0))),
        "terminal_weight": float(stage.get("terminal_weight", acfg.get("terminal_weight", 1.0))),
        "reward_mode": str(stage.get("reward_mode", acfg.get("reward_mode", "sequence"))),
        "dense_step_weight": float(stage.get("dense_step_weight", acfg.get("dense_step_weight", 1.0))),
        "partial_credit": bool(stage.get("partial_credit", acfg.get("partial_credit", True))),
        "normalize_dense_reward": bool(
            stage.get("normalize_dense_reward", acfg.get("normalize_dense_reward", True))
        ),
    }


def get_curriculum_stage(cfg: dict, step: int) -> tuple[int, dict]:
    stages = cfg.get("curriculum", {}).get("stages", [])

    if not stages:
        return 0, {}

    for i, stage in enumerate(stages):
        start = int(stage.get("start_step", 1))
        end = int(stage.get("end_step", cfg.get("train", {}).get("steps", 1000)))

        if start <= step <= end:
            return i + 1, stage

    if step < int(stages[0].get("start_step", 1)):
        return 1, stages[0]

    return len(stages), stages[-1]


def save_checkpoint(
    run_dir: Path,
    cfg: dict,
    model,
    tokenizer: Tokenizer,
    filename: str = "checkpoint.pt",
) -> Path:
    ckpt = {
        "config": cfg,
        "model": model.state_dict(),
        "vocab": tokenizer.id_to_token,
    }

    ckpt_path = run_dir / filename
    torch.save(ckpt, ckpt_path)

    return ckpt_path


def log_train_metrics(
    run_dir: Path,
    history: list[dict],
    step: int,
    alg: str,
    metrics: dict,
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

    loss = F.cross_entropy(
        logits[action_mask[:, 1:]],
        targets[action_mask[:, 1:]],
    )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()

    preds = logits[action_mask[:, 1:]].argmax(dim=-1)
    acc = (preds == targets[action_mask[:, 1:]]).float().mean()

    return {
        "loss": float(loss.item()),
        "token_acc": float(acc.item()),
    }


def train(cfg: dict, init_from_checkpoint: Optional[str] = None) -> Path:
    rng = seed_everything(cfg.get("seed", 0))

    device = device_auto()
    tokenizer = Tokenizer.default()
    task = build_task(cfg["task"]["name"])

    if init_from_checkpoint:
        ckpt_path = Path(init_from_checkpoint)

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=device)

        if "model" not in ckpt:
            raise ValueError(f"Checkpoint at {ckpt_path} does not contain 'model' weights")

        checkpoint_cfg = ckpt.get("config", {})

        if checkpoint_cfg.get("model"):
            current_model_cfg = copy.deepcopy(cfg.get("model", {}))
            checkpoint_model_cfg = checkpoint_cfg.get("model", {})

            if current_model_cfg != checkpoint_model_cfg:
                print(
                    "Checkpoint model architecture differs from current config; "
                    "using the checkpoint's model settings for warm start.",
                    flush=True,
                )

                cfg = copy.deepcopy(cfg)
                cfg["model"] = {**current_model_cfg, **checkpoint_model_cfg}

    model = build_model(cfg, tokenizer).to(device)

    if init_from_checkpoint:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded pretrained weights from {ckpt_path}", flush=True)

    alg = cfg["algorithm"]["name"]

    run_dir = resolve_run_dir(cfg, task.name, alg)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"].get("lr", 3e-4),
    )

    (run_dir / "train_metrics.jsonl").write_text("", encoding="utf-8")
    save_json([], run_dir / "train_metrics.json")

    steps = cfg["train"].get("steps", 1000)
    batch_size = cfg["train"].get("batch_size", 64)

    base_min_len = cfg["train"].get("train_min_len", TRAIN_MIN_LEN)
    base_max_len = cfg["train"].get("train_max_len", TRAIN_MAX_LEN)
    train_len_cap = cfg["train"].get("train_max_len_cap", 500)

    if base_min_len < 1 or base_max_len > train_len_cap:
        raise ValueError(f"Training lengths must stay within 1..{train_len_cap}.")

    gen = cfg.get("generation", {})
    base_max_new = gen.get("max_new_tokens", 520)
    base_temp = gen.get("temperature", 1.0)

    eval_gen = cfg.get("eval_generation", {})
    eval_max_new = eval_gen.get("max_new_tokens", base_max_new)
    eval_temp = eval_gen.get("temperature", 0.8)
    eval_max_length = eval_gen.get("max_length", None)

    progress = trange(1, steps + 1, desc=alg)

    last_metrics = {}
    train_history = []
    started_at = time.time()

    log_every = cfg["train"].get("log_every", 50)
    eval_every = cfg["train"].get("eval_every", 200)
    save_every = cfg["train"].get("save_every", 0)

    for step in progress:
        model.train()

        stage_idx, stage = get_curriculum_stage(cfg, step)

        step_min_len = int(stage.get("train_min_len", base_min_len))
        step_max_len = int(stage.get("train_max_len", base_max_len))
        step_max_new = int(stage.get("max_new_tokens", base_max_new))
        step_temp = float(stage.get("temperature", base_temp))

        if step_min_len < 1 or step_max_len > train_len_cap or step_min_len > step_max_len:
            raise ValueError(
                f"Invalid training length range at step {step}: "
                f"{step_min_len}..{step_max_len}. "
                f"Allowed range is 1..{train_len_cap}."
            )

        if alg == "sft":
            examples = sample_examples(
                task,
                rng,
                batch_size,
                step_min_len,
                step_max_len,
            )

            last_metrics = sft_step(
                model,
                optimizer,
                tokenizer,
                examples,
                device,
            )

        elif alg.startswith("ppo_"):
            acfg = cfg["algorithm"]

            reward_kwargs = resolve_reward_kwargs(acfg, stage)

            rollouts = [
                collect_rollout(
                    model=model,
                    tokenizer=tokenizer,
                    task=task,
                    example=ex,
                    algorithm_name=alg,
                    max_new_tokens=step_max_new,
                    temperature=step_temp,
                    device=device,
                    **reward_kwargs,
                )
                for ex in sample_examples(
                    task,
                    rng,
                    acfg.get("rollout_batch_size", batch_size),
                    step_min_len,
                    step_max_len,
                )
            ]

            last_metrics = ppo_update(
                model=model,
                optimizer=optimizer,
                tokenizer=tokenizer,
                rollouts=rollouts,
                clip_eps=acfg.get("clip_eps", 0.2),
                value_coef=acfg.get("value_coef", 0.5),
                entropy_coef=acfg.get("entropy_coef", 0.01),
                epochs=acfg.get("ppo_epochs", 4),
                device=device,
            )

        elif alg.startswith("grpo_"):
            acfg = cfg["algorithm"]

            reward_kwargs = resolve_reward_kwargs(acfg, stage)

            groups = []

            for ex in sample_examples(
                task,
                rng,
                acfg.get("groups_per_step", batch_size),
                step_min_len,
                step_max_len,
            ):
                group = []

                for _ in range(acfg.get("samples_per_prompt", 4)):
                    group.append(
                        collect_rollout(
                            model=model,
                            tokenizer=tokenizer,
                            task=task,
                            example=ex,
                            algorithm_name=alg,
                            max_new_tokens=step_max_new,
                            temperature=step_temp,
                            device=device,
                            **reward_kwargs,
                        )
                    )

                groups.append(group)

            last_metrics = grpo_update(
                model=model,
                optimizer=optimizer,
                tokenizer=tokenizer,
                grouped_rollouts=groups,
                clip_eps=acfg.get("clip_eps", 0.2),
                entropy_coef=acfg.get("entropy_coef", 0.01),
                device=device,
            )

        else:
            raise ValueError(f"Unknown algorithm: {alg}")

        last_metrics = {
            **last_metrics,
            "curriculum_stage": float(stage_idx),
            "train_min_len": float(step_min_len),
            "train_max_len": float(step_max_len),
            "generation_max_new_tokens": float(step_max_new),
            "generation_temperature": float(step_temp),
        }

        progress.set_postfix(last_metrics)

        if log_every and step % log_every == 0:
            log_train_metrics(
                run_dir=run_dir,
                history=train_history,
                step=step,
                alg=alg,
                metrics=last_metrics,
                started_at=started_at,
            )

        if save_every and step % save_every == 0:
            save_checkpoint(run_dir, cfg, model, tokenizer, f"checkpoint_step{step}.pt")
            save_checkpoint(run_dir, cfg, model, tokenizer, "checkpoint.pt")

        if eval_every and step % eval_every == 0:
            metrics = evaluate_buckets(
                model=model,
                tokenizer=tokenizer,
                task=task,
                samples_per_length=1,
                max_new_tokens=eval_max_new,
                device=device,
                max_length=eval_max_length,
                temperature=eval_temp,
            )

            save_json(
                {
                    "step": step,
                    "train": last_metrics,
                    "eval": metrics,
                },
                run_dir / f"metrics_step{step}.json",
            )

    ckpt_path = save_checkpoint(
        run_dir=run_dir,
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        filename="checkpoint.pt",
    )

    if log_every and steps % log_every != 0:
        log_train_metrics(
            run_dir=run_dir,
            history=train_history,
            step=steps,
            alg=alg,
            metrics=last_metrics,
            started_at=started_at,
        )

    save_json(
        {
            "final_train_metrics": last_metrics,
            "train_log_every": log_every,
            "checkpoint": str(ckpt_path),
        },
        run_dir / "summary.json",
    )

    return ckpt_path


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--log-every", type=int)
    parser.add_argument(
        "--init-from-checkpoint",
        type=str,
        help="Load model weights from an existing checkpoint before training",
    )

    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.steps is not None:
        cfg["train"]["steps"] = args.steps

    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size

    if args.log_every is not None:
        cfg["train"]["log_every"] = args.log_every

    path = train(cfg, init_from_checkpoint=args.init_from_checkpoint)

    print(path)


if __name__ == "__main__":
    main()