from __future__ import annotations

import random
from typing import Iterable

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
            sampled, _, _ = model.generate(prompt, tokenizer.eos_id, max_new_tokens, 0.8, device)
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
) -> dict[str, dict[str, float]]:
    out = {
        "train_1_40": evaluate_lengths(model, tokenizer, task, range(1, 41), samples_per_length, max_new_tokens, device, 11)
    }
    for lo, hi in OOD_BUCKETS:
        out[f"ood_{lo}_{hi}"] = evaluate_lengths(
            model, tokenizer, task, range(lo, hi + 1), samples_per_length, max_new_tokens, device, 1000 + lo
        )
    return out
