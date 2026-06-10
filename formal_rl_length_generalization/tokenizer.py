from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch

from .tasks import all_task_tokens


@dataclass
class Tokenizer:
    token_to_id: dict[str, int]
    id_to_token: list[str]

    @classmethod
    def default(cls) -> "Tokenizer":
        vocab = all_task_tokens()
        return cls({tok: i for i, tok in enumerate(vocab)}, vocab)

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<eos>"]

    def encode(self, tokens: Sequence[str], add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = [self.token_to_id[t] for t in tokens]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> List[str]:
        toks = [self.id_to_token[int(i)] for i in ids]
        if skip_special:
            toks = [t for t in toks if t not in {"<pad>", "<bos>", "<eos>"}]
        return toks

    def pad_batch(self, rows: Sequence[Sequence[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = max(len(r) for r in rows)
        x = torch.full((len(rows), max_len), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(rows), max_len), dtype=torch.bool)
        for i, row in enumerate(rows):
            x[i, : len(row)] = torch.tensor(row, dtype=torch.long)
            mask[i, : len(row)] = True
        return x, mask
