from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class EventEncoderConfig:
    vocab_size: int
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    ffn_dim: int = 1024
    dropout: float = 0.1
    max_seq_len: int = 151
    time_gap_vocab_size: int = 10
    session_vocab_size: int = 2
    pad_token_id: int = 0
    next_pooling: str = "cls"


class EventTransformerEncoder(nn.Module):
    def __init__(self, config: EventEncoderConfig) -> None:
        super().__init__()
        self.config = config

        self.event_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=config.pad_token_id,
        )
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.time_gap_embedding = nn.Embedding(config.time_gap_vocab_size, config.d_model)
        self.session_embedding = nn.Embedding(config.session_vocab_size, config.d_model)
        self.input_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        if config.next_pooling not in {"cls", "last", "cls_last_concat"}:
            raise ValueError(f"Unsupported next_pooling: {config.next_pooling}")
        self.next_pooling = config.next_pooling

        self.mlm_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.vocab_size),
        )
        next_head_dim = config.d_model * 2 if self.next_pooling == "cls_last_concat" else config.d_model
        self.next_head = nn.Linear(next_head_dim, config.vocab_size)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].fill_(0)

    def forward(
        self,
        input_ids: torch.Tensor,
        time_gap_ids: torch.Tensor,
        session_flags: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)

        hidden = (
            self.event_embedding(input_ids)
            + self.position_embedding(positions)
            + self.time_gap_embedding(time_gap_ids)
            + self.session_embedding(session_flags)
        )
        hidden = self.dropout(self.input_norm(hidden))

        encoded = self.encoder(hidden, src_key_padding_mask=~attention_mask)
        cls_embedding = encoded[:, 0]
        lengths = attention_mask.long().sum(dim=1).clamp_min(1) - 1
        gather_idx = lengths.view(-1, 1, 1).expand(-1, 1, encoded.size(-1))
        last_embedding = encoded.gather(1, gather_idx).squeeze(1)
        if self.next_pooling == "cls":
            next_embedding = cls_embedding
        elif self.next_pooling == "last":
            next_embedding = last_embedding
        else:
            next_embedding = torch.cat([cls_embedding, last_embedding], dim=-1)

        return {
            "last_hidden_state": encoded,
            "cls_embedding": cls_embedding,
            "last_embedding": last_embedding,
            "sequence_embedding": next_embedding,
            "next_logits": self.next_head(next_embedding),
        }
