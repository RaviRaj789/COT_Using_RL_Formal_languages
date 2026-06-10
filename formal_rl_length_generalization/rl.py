from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F

from .model import CausalTransformerPolicy
from .tasks import Example, FormalTask
from .tokenizer import Tokenizer


@dataclass
class Rollout:
    example: Example
    row: list[int]
    action_mask: list[bool]
    old_logps: list[float]
    rewards: list[float]
    generated_tokens: list[str]
    process_match: float
    output_match: float
    exact_match: float
    cot_tokens: int


def reward_scalar(name: str, process: float, terminal: float, process_weight: float = 1.0, terminal_weight: float = 1.0) -> float:
    if name.endswith("process_terminal"):
        return process_weight * process + terminal_weight * terminal
    if name.endswith("process"):
        return process
    if name.endswith("terminal"):
        return terminal
    raise ValueError(f"Unknown RL reward mode for {name}")


def collect_rollout(
    model: CausalTransformerPolicy,
    tokenizer: Tokenizer,
    task: FormalTask,
    example: Example,
    algorithm_name: str,
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
    process_weight: float = 1.0,
    terminal_weight: float = 1.0,
) -> Rollout:
    prompt_ids = tokenizer.encode(example.prompt_tokens, add_bos=True)
    sampled, logps, _ = model.generate(prompt_ids, tokenizer.eos_id, max_new_tokens, temperature, device)
    decoded = tokenizer.decode(sampled)
    reward = task.reward(example, decoded)
    score = reward_scalar(algorithm_name, reward.process, reward.terminal, process_weight, terminal_weight)
    row = prompt_ids + sampled
    action_mask = [False] * len(prompt_ids) + [True] * len(sampled)
    cot_tokens = decoded.index("FINAL") if "FINAL" in decoded else len(decoded)
    exact_match = float(decoded[: len(example.target_tokens)] == example.target_tokens)
    return Rollout(
        example,
        row,
        action_mask,
        logps,
        [score] * len(sampled),
        decoded,
        reward.process,
        reward.terminal,
        exact_match,
        cot_tokens,
    )


def rollout_stats(rollouts: List[Rollout]) -> dict[str, float]:
    if not rollouts:
        return {}
    n = len(rollouts)
    generated_tokens = [len(r.generated_tokens) for r in rollouts]
    cot_tokens = [r.cot_tokens for r in rollouts]
    process_matches = [r.process_match for r in rollouts]
    output_matches = [r.output_match for r in rollouts]
    exact_matches = [r.exact_match for r in rollouts]
    return {
        "rollout_count": float(n),
        "generated_tokens_mean": sum(generated_tokens) / n,
        "cot_tokens_mean": sum(cot_tokens) / n,
        "cot_tokens_total": float(sum(cot_tokens)),
        "cot_match_accuracy": sum(process_matches) / n,
        "cot_match_percent": 100.0 * sum(process_matches) / n,
        "process_match_accuracy": sum(process_matches) / n,
        "process_match_percent": 100.0 * sum(process_matches) / n,
        "output_match_accuracy": sum(output_matches) / n,
        "output_match_percent": 100.0 * sum(output_matches) / n,
        "exact_match_accuracy": sum(exact_matches) / n,
        "exact_match_percent": 100.0 * sum(exact_matches) / n,
    }


def _pad_rollouts(tokenizer: Tokenizer, rollouts: List[Rollout], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rows, row_mask = tokenizer.pad_batch([r.row for r in rollouts])
    action_mask, _ = tokenizer.pad_batch([[int(x) for x in r.action_mask] for r in rollouts])
    old_logps = torch.cat([torch.tensor(r.old_logps, dtype=torch.float32) for r in rollouts])
    rewards = torch.cat([torch.tensor(r.rewards, dtype=torch.float32) for r in rollouts])
    return rows.to(device), action_mask.bool().to(device), old_logps.to(device), rewards.to(device)


def ppo_update(
    model: CausalTransformerPolicy,
    optimizer: torch.optim.Optimizer,
    tokenizer: Tokenizer,
    rollouts: List[Rollout],
    clip_eps: float,
    value_coef: float,
    entropy_coef: float,
    epochs: int,
    device: torch.device,
) -> dict[str, float]:
    rows, action_mask, old_logps, returns = _pad_rollouts(tokenizer, rollouts, device)
    metrics = {}
    for _ in range(epochs):
        logits, values = model(rows[:, :-1])
        targets = rows[:, 1:]
        act = action_mask[:, 1:]
        logp_all = torch.log_softmax(logits, dim=-1)
        logps = logp_all.gather(-1, targets.unsqueeze(-1)).squeeze(-1)[act]
        vals = values[act]
        advantages = returns - vals.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std().clamp_min(1e-6))
        ratio = torch.exp(logps - old_logps)
        policy_loss = -torch.minimum(ratio * advantages, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages).mean()
        value_loss = F.mse_loss(vals, returns)
        entropy = -(logp_all.exp() * logp_all).sum(dim=-1)[act].mean()
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        metrics = {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
            "reward": float(returns.mean().item()),
        }
    metrics.update(rollout_stats(rollouts))
    return metrics


def grpo_update(
    model: CausalTransformerPolicy,
    optimizer: torch.optim.Optimizer,
    tokenizer: Tokenizer,
    grouped_rollouts: List[List[Rollout]],
    clip_eps: float,
    entropy_coef: float,
    device: torch.device,
) -> dict[str, float]:
    flattened: List[Rollout] = []
    advantages = []
    for group in grouped_rollouts:
        scores = torch.tensor([sum(r.rewards) / max(1, len(r.rewards)) for r in group], dtype=torch.float32)
        group_adv = (scores - scores.mean()) / scores.std().clamp_min(1e-6)
        for rollout, adv in zip(group, group_adv):
            flattened.append(rollout)
            advantages.extend([float(adv.item())] * len(rollout.old_logps))
    rows, action_mask, old_logps, _ = _pad_rollouts(tokenizer, flattened, device)
    adv_t = torch.tensor(advantages, dtype=torch.float32, device=device)
    logits, _ = model(rows[:, :-1])
    targets = rows[:, 1:]
    act = action_mask[:, 1:]
    logp_all = torch.log_softmax(logits, dim=-1)
    logps = logp_all.gather(-1, targets.unsqueeze(-1)).squeeze(-1)[act]
    ratio = torch.exp(logps - old_logps)
    policy_loss = -torch.minimum(ratio * adv_t, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_t).mean()
    entropy = -(logp_all.exp() * logp_all).sum(dim=-1)[act].mean()
    loss = policy_loss - entropy_coef * entropy
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    rewards = [sum(r.rewards) / max(1, len(r.rewards)) for r in flattened]
    metrics = {
        "loss": float(loss.item()),
        "policy_loss": float(policy_loss.item()),
        "entropy": float(entropy.item()),
        "reward": float(torch.tensor(rewards).mean().item()),
    }
    metrics.update(rollout_stats(flattened))
    return metrics
