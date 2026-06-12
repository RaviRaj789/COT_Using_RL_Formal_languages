from __future__ import annotations

import random
from typing import Iterable, Optional

import torch

from .model import CausalTransformerPolicy
from .tasks import FormalTask, OOD_BUCKETS
from .tokenizer import Tokenizer


@torch.no_grad()
def evaluate_lengths(
    model: CausalTransformerPolicy,
    tokenizer: Tokenizer,
    task: FormalTask,
    lengths: Iterable[int],
    samples_per_length: int,
    max_new_tokens: int,
    device: torch.device,
    seed: int = 0,
    temperature: float = 0.8,
) -> dict[str, float]:
    rng = random.Random(seed)

    process_scores = []
    terminal_scores = []
    exact_scores = []

    model.eval()

    for n in lengths:
        for _ in range(samples_per_length):
            ex = task.sample(n, rng)

            prompt = tokenizer.encode(ex.prompt_tokens, add_bos=True)

            sampled, _, _ = model.generate(
                prompt,
                tokenizer.eos_id,
                max_new_tokens,
                temperature,
                device,
            )

            decoded = tokenizer.decode(sampled)
            reward = task.reward(ex, decoded)

            process_scores.append(reward.process)
            terminal_scores.append(reward.terminal)
            exact_scores.append(float(decoded[: len(ex.target_tokens)] == ex.target_tokens))

    return {
        "process": sum(process_scores) / max(1, len(process_scores)),
        "terminal": sum(terminal_scores) / max(1, len(terminal_scores)),
        "exact": sum(exact_scores) / max(1, len(exact_scores)),
    }


def evaluate_buckets(
    model: CausalTransformerPolicy,
    tokenizer: Tokenizer,
    task: FormalTask,
    samples_per_length: int,
    max_new_tokens: int,
    device: torch.device,
    max_length: Optional[int] = None,
    temperature: float = 0.8,
) -> dict[str, dict[str, float]]:
    """
    Evaluate train and OOD buckets.

    max_length is useful for cheap periodic eval.
    Example:
      max_length=80 evaluates only train_1_40 and ood_41_80.
      max_length=500 evaluates all buckets.
    """

    out: dict[str, dict[str, float]] = {}

    train_hi = 40 if max_length is None else min(40, max_length)

    if train_hi >= 1:
        out[f"train_1_{train_hi}"] = evaluate_lengths(
            model=model,
            tokenizer=tokenizer,
            task=task,
            lengths=range(1, train_hi + 1),
            samples_per_length=samples_per_length,
            max_new_tokens=max_new_tokens,
            device=device,
            seed=11,
            temperature=temperature,
        )

    for lo, hi in OOD_BUCKETS:
        if max_length is not None and lo > max_length:
            break

        bucket_hi = hi if max_length is None else min(hi, max_length)

        if bucket_hi < lo:
            continue

        out[f"ood_{lo}_{bucket_hi}"] = evaluate_lengths(
            model=model,
            tokenizer=tokenizer,
            task=task,
            lengths=range(lo, bucket_hi + 1),
            samples_per_length=samples_per_length,
            max_new_tokens=max_new_tokens,
            device=device,
            seed=1000 + lo,
            temperature=temperature,
        )

    return out