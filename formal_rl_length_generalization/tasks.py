from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence


TRAIN_MIN_LEN = 1
TRAIN_MAX_LEN = 40
OOD_BUCKETS = [(41, 80), (81, 160), (161, 320), (321, 500)]
BALANCE_CAP = 500


@dataclass(frozen=True)
class Example:
    task_name: str
    input_tokens: List[str]
    trace_tokens: List[str]
    final_token: str

    @property
    def target_tokens(self) -> List[str]:
        return self.trace_tokens + ["FINAL", self.final_token]

    @property
    def prompt_tokens(self) -> List[str]:
        return ["TASK", self.task_name, "INPUT"] + self.input_tokens + ["TRACE"]


@dataclass(frozen=True)
class Reward:
    process: float
    terminal: float


class FormalTask(Protocol):
    name: str

    def sample(self, n: int, rng: random.Random) -> Example:
        ...

    def reward(self, example: Example, generated: Sequence[str]) -> Reward:
        ...

    def tokens(self) -> List[str]:
        ...


def _final_after_marker(tokens: Sequence[str]) -> str:
    if "FINAL" not in tokens:
        return ""
    idx = len(tokens) - 1 - list(reversed(tokens)).index("FINAL")
    return tokens[idx + 1] if idx + 1 < len(tokens) else ""


def _prefix_before_final(tokens: Sequence[str]) -> List[str]:
    if "FINAL" not in tokens:
        return list(tokens)
    return list(tokens[: tokens.index("FINAL")])


class ParityTask:
    name = "parity"

    def sample(self, n: int, rng: random.Random) -> Example:
        xs = [str(rng.randint(0, 1)) for _ in range(n)]
        p = 0
        trace = []
        for x in xs:
            p ^= int(x)
            trace.append(f"P{p}")
        return Example(self.name, xs, trace, f"P{p}")

    def reward(self, example: Example, generated: Sequence[str]) -> Reward:
        pred_trace = _prefix_before_final(generated)
        hits = sum(a == b for a, b in zip(pred_trace, example.trace_tokens))
        process = hits / max(1, len(example.trace_tokens))
        terminal = float(_final_after_marker(generated) == example.final_token)
        return Reward(process, terminal)

    def tokens(self) -> List[str]:
        return ["0", "1", "P0", "P1"]


class ModularCountingTask:
    def __init__(self, k: int):
        self.k = k
        self.name = f"mod_{k}"

    def sample(self, n: int, rng: random.Random) -> Example:
        xs = [str(rng.randint(0, 1)) for _ in range(n)]
        c = 0
        trace = []
        for x in xs:
            c = (c + int(x)) % self.k
            trace.append(f"C{c}")
        return Example(self.name, xs, trace, "ACCEPT" if c == 0 else "REJECT")

    def reward(self, example: Example, generated: Sequence[str]) -> Reward:
        pred_trace = _prefix_before_final(generated)
        hits = sum(a == b for a, b in zip(pred_trace, example.trace_tokens))
        process = hits / max(1, len(example.trace_tokens))
        terminal = float(_final_after_marker(generated) == example.final_token)
        return Reward(process, terminal)

    def tokens(self) -> List[str]:
        return ["0", "1", "ACCEPT", "REJECT"] + [f"C{i}" for i in range(self.k)]


class AStarBStarTask:
    name = "a_star_b_star"

    def sample(self, n: int, rng: random.Random) -> Example:
        xs = [rng.choice(["a", "b"]) for _ in range(n)]
        state = 0
        trace = []
        for x in xs:
            if state == 0 and x == "a":
                state = 0
            elif state == 0 and x == "b":
                state = 1
            elif state == 1 and x == "b":
                state = 1
            else:
                state = 2
            trace.append(f"S{state}")
        return Example(self.name, xs, trace, "ACCEPT" if state in {0, 1} else "REJECT")

    def reward(self, example: Example, generated: Sequence[str]) -> Reward:
        pred_trace = _prefix_before_final(generated)
        hits = sum(a == b for a, b in zip(pred_trace, example.trace_tokens))
        process = hits / max(1, len(example.trace_tokens))
        terminal = float(_final_after_marker(generated) == example.final_token)
        return Reward(process, terminal)

    def tokens(self) -> List[str]:
        return ["a", "b", "S0", "S1", "S2", "ACCEPT", "REJECT"]


class ANBNTask:
    name = "an_bn"

    def sample(self, n: int, rng: random.Random) -> Example:
        if rng.random() < 0.5:
            m = n // 2
            if n % 2 == 0 and m > 0:
                xs = ["a"] * m + ["b"] * m
            else:
                xs = [rng.choice(["a", "b"]) for _ in range(n)]
        else:
            xs = [rng.choice(["a", "b"]) for _ in range(n)]
        phase = "A_PHASE"
        count_a = 0
        count_b = 0
        dead = False
        trace = []
        for x in xs:
            if x == "a":
                if phase == "B_PHASE":
                    dead = True
                    phase = "DEAD"
                elif not dead:
                    count_a += 1
            elif x == "b":
                if not dead:
                    phase = "B_PHASE"
                    count_b += 1
            bal = max(-BALANCE_CAP, min(BALANCE_CAP, count_a - count_b))
            trace.extend([phase, f"BAL_{bal}"])
        accept = (not dead) and count_a == count_b and (phase in {"A_PHASE", "B_PHASE"})
        return Example(self.name, xs, trace, "ACCEPT" if accept else "REJECT")

    def reward(self, example: Example, generated: Sequence[str]) -> Reward:
        pred = _prefix_before_final(generated)
        total = 0.0
        steps = max(1, len(example.trace_tokens) // 2)
        for i in range(steps):
            phase_true = example.trace_tokens[2 * i]
            bal_true = _parse_balance(example.trace_tokens[2 * i + 1])
            phase_pred = pred[2 * i] if 2 * i < len(pred) else ""
            bal_pred = _parse_balance(pred[2 * i + 1]) if 2 * i + 1 < len(pred) else None
            phase_r = float(phase_pred == phase_true)
            bal_r = 0.0 if bal_pred is None else 1.0 - min(abs(bal_pred - bal_true), BALANCE_CAP) / BALANCE_CAP
            total += 0.5 * phase_r + 0.5 * bal_r
        terminal = float(_final_after_marker(generated) == example.final_token)
        return Reward(total / steps, terminal)

    def tokens(self) -> List[str]:
        return ["a", "b", "A_PHASE", "B_PHASE", "DEAD", "ACCEPT", "REJECT"] + [
            f"BAL_{i}" for i in range(-BALANCE_CAP, BALANCE_CAP + 1)
        ]


def _parse_balance(token: str) -> Optional[int]:
    if not token.startswith("BAL_"):
        return None
    try:
        return int(token[4:])
    except ValueError:
        return None


def build_task(name: str) -> FormalTask:
    if name == "parity":
        return ParityTask()
    if name == "mod_3":
        return ModularCountingTask(3)
    if name == "mod_5":
        return ModularCountingTask(5)
    if name == "a_star_b_star":
        return AStarBStarTask()
    if name in {"an_bn", "a_n_b_n"}:
        return ANBNTask()
    raise ValueError(f"Unknown task: {name}")


def all_task_tokens() -> List[str]:
    tokens = ["<pad>", "<bos>", "<eos>", "TASK", "INPUT", "TRACE", "FINAL"]
    for task in [ParityTask(), ModularCountingTask(3), ModularCountingTask(5), AStarBStarTask(), ANBNTask()]:
        tokens.extend([task.name])
        tokens.extend(task.tokens())
    return sorted(set(tokens), key=tokens.index)
