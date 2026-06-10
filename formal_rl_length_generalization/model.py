from __future__ import annotations

import torch
import torch.nn as nn


class CausalTransformerPolicy(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_model)
        self.policy_head = nn.Linear(d_model, vocab_size)
        self.value_head = nn.Linear(d_model, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len = input_ids.shape
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        causal = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device), diagonal=1)
        padding = input_ids.eq(self.pad_id)
        h = self.blocks(x, mask=causal, src_key_padding_mask=padding)
        h = self.ln(h)
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: list[int],
        eos_id: int,
        max_new_tokens: int,
        temperature: float = 1.0,
        device: torch.device | str = "cpu",
    ) -> tuple[list[int], list[float], list[float]]:
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        sampled: list[int] = []
        logps: list[float] = []
        values: list[float] = []
        for _ in range(max_new_tokens):
            logits, vals = self(ids)
            next_logits = logits[0, -1] / max(temperature, 1e-6)
            probs = torch.softmax(next_logits, dim=-1)
            token = torch.multinomial(probs, 1)
            logp = torch.log(probs[token] + 1e-12)
            sampled.append(int(token.item()))
            logps.append(float(logp.item()))
            values.append(float(vals[0, -1].item()))
            ids = torch.cat([ids, token.view(1, 1)], dim=1)
            if int(token.item()) == eos_id:
                break
        return sampled, logps, values


def sequence_logprobs(model: CausalTransformerPolicy, rows: torch.Tensor, action_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logits, values = model(rows[:, :-1])
    targets = rows[:, 1:]
    logp = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return logp[action_mask[:, 1:]], values[action_mask[:, 1:]]
