from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, List, Optional, Protocol, Sequence, Tuple


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
    """Formal task interface.

    The ``step_*`` / ``*_state`` / ``*_transition`` methods below define a small
    automaton schema (state, symbol -> next state) that is independent of the
    ``reward`` method above. They are used only by dense_reward.py to build
    per-step partial-credit scores and are what a new task must implement to
    plug into dense process rewards.
    """

    name: str

    def sample(self, n: int, rng: random.Random) -> Example:
        ...

    def reward(self, example: Example, generated: Sequence[str]) -> Reward:
        ...

    def tokens(self) -> List[str]:
        ...

    def step_size(self) -> int:
        """Number of trace tokens that make up one reasoning step."""
        ...

    def initial_state(self) -> Any:
        ...

    def apply_transition(self, state: Any, symbol: str) -> Any:
        """Oracle transition function: (state, input symbol) -> next state."""
        ...

    def state_to_step_tokens(self, state: Any) -> Tuple[str, ...]:
        """Render a state as the step tokens an oracle trace would emit for it."""
        ...

    def parse_step_state(self, step_tokens: Sequence[str]) -> Optional[Any]:
        """Parse a generated step's tokens back into a state, or None if malformed."""
        ...

    def step_match_score(self, oracle_step: Sequence[str], gen_step: Sequence[str]) -> float:
        """Similarity in [0, 1] between an oracle step and a generated step."""
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

    def step_size(self) -> int:
        return 1

    def initial_state(self) -> Any:
        return 0

    def apply_transition(self, state: Any, symbol: str) -> Any:
        return int(state) ^ int(symbol)

    def state_to_step_tokens(self, state: Any) -> Tuple[str, ...]:
        return (f"P{state}",)

    def parse_step_state(self, step_tokens: Sequence[str]) -> Optional[Any]:
        if len(step_tokens) != 1 or not step_tokens[0].startswith("P"):
            return None
        try:
            value = int(step_tokens[0][1:])
        except ValueError:
            return None
        return value if value in (0, 1) else None

    def step_match_score(self, oracle_step: Sequence[str], gen_step: Sequence[str]) -> float:
        return float(tuple(oracle_step) == tuple(gen_step))


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

    def step_size(self) -> int:
        return 1

    def initial_state(self) -> Any:
        return 0

    def apply_transition(self, state: Any, symbol: str) -> Any:
        return (int(state) + int(symbol)) % self.k

    def state_to_step_tokens(self, state: Any) -> Tuple[str, ...]:
        return (f"C{state}",)

    def parse_step_state(self, step_tokens: Sequence[str]) -> Optional[Any]:
        if len(step_tokens) != 1 or not step_tokens[0].startswith("C"):
            return None
        try:
            value = int(step_tokens[0][1:])
        except ValueError:
            return None
        return value if 0 <= value < self.k else None

    def step_match_score(self, oracle_step: Sequence[str], gen_step: Sequence[str]) -> float:
        return float(tuple(oracle_step) == tuple(gen_step))


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

    def step_size(self) -> int:
        return 1

    def initial_state(self) -> Any:
        return 0

    def apply_transition(self, state: Any, symbol: str) -> Any:
        if state == 0 and symbol == "a":
            return 0
        if state == 0 and symbol == "b":
            return 1
        if state == 1 and symbol == "b":
            return 1
        return 2

    def state_to_step_tokens(self, state: Any) -> Tuple[str, ...]:
        return (f"S{state}",)

    def parse_step_state(self, step_tokens: Sequence[str]) -> Optional[Any]:
        if len(step_tokens) != 1 or not step_tokens[0].startswith("S"):
            return None
        try:
            value = int(step_tokens[0][1:])
        except ValueError:
            return None
        return value if value in (0, 1, 2) else None

    def step_match_score(self, oracle_step: Sequence[str], gen_step: Sequence[str]) -> float:
        return float(tuple(oracle_step) == tuple(gen_step))


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

    def step_size(self) -> int:
        return 2

    def initial_state(self) -> Any:
        return ("A_PHASE", 0)

    def apply_transition(self, state: Any, symbol: str) -> Any:
        phase, bal = state
        if symbol == "a":
            if phase == "B_PHASE":
                phase = "DEAD"
            elif phase != "DEAD":
                bal = _clamp_balance(bal + 1)
        elif symbol == "b":
            if phase != "DEAD":
                phase = "B_PHASE"
                bal = _clamp_balance(bal - 1)
        return (phase, bal)

    def state_to_step_tokens(self, state: Any) -> Tuple[str, ...]:
        phase, bal = state
        return (phase, f"BAL_{bal}")

    def parse_step_state(self, step_tokens: Sequence[str]) -> Optional[Any]:
        if len(step_tokens) != 2:
            return None
        phase, bal_token = step_tokens
        if phase not in {"A_PHASE", "B_PHASE", "DEAD"}:
            return None
        bal = _parse_balance(bal_token)
        return None if bal is None else (phase, bal)

    def step_match_score(self, oracle_step: Sequence[str], gen_step: Sequence[str]) -> float:
        if len(gen_step) != 2:
            return 0.0
        phase_true, bal_true_token = oracle_step
        phase_pred, bal_pred_token = gen_step
        bal_true = _parse_balance(bal_true_token)
        bal_pred = _parse_balance(bal_pred_token)
        phase_r = float(phase_pred == phase_true)
        if bal_true is None or bal_pred is None:
            bal_r = 0.0
        else:
            bal_r = 1.0 - min(abs(bal_pred - bal_true), BALANCE_CAP) / BALANCE_CAP
        return 0.5 * phase_r + 0.5 * bal_r


def _clamp_balance(value: int) -> int:
    return max(-BALANCE_CAP, min(BALANCE_CAP, value))


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
