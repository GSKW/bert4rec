from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SpecialTokenIds:
    pad: int
    unk: int
    mask: int
    cls: int


class PrefixDataset(Dataset):
    def __init__(self, parquet_path: str | Path, max_rows: int | None = None) -> None:
        import pandas as pd

        columns = [
            "event_token_ids",
            "time_gap_ids",
            "session_flags",
            "next_event_token_id",
            "prefix_len",
        ]
        frame = pd.read_parquet(parquet_path, columns=columns)
        if max_rows is not None:
            frame = frame.head(max_rows)

        frame = frame[frame["next_event_token_id"].notna()].reset_index(drop=True)
        self.event_token_ids = frame["event_token_ids"].tolist()
        self.time_gap_ids = frame["time_gap_ids"].tolist()
        self.session_flags = frame["session_flags"].tolist()
        self.next_event_token_ids = frame["next_event_token_id"].astype("int64").tolist()
        self.prefix_lens = frame["prefix_len"].astype("int64").tolist()

    def __len__(self) -> int:
        return len(self.next_event_token_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "event_token_ids": self.event_token_ids[idx],
            "time_gap_ids": self.time_gap_ids[idx],
            "session_flags": self.session_flags[idx],
            "next_event_token_id": self.next_event_token_ids[idx],
            "prefix_len": self.prefix_lens[idx],
        }


class PrefixCollator:
    def __init__(
        self,
        special_token_ids: SpecialTokenIds,
        max_seq_len: int,
        mlm_probability: float = 0.15,
        mask_token_probability: float = 0.8,
        random_token_probability: float = 0.1,
        vocab_size: int | None = None,
        train: bool = True,
    ) -> None:
        self.special = special_token_ids
        self.max_seq_len = max_seq_len
        self.mlm_probability = mlm_probability
        self.mask_token_probability = mask_token_probability
        self.random_token_probability = random_token_probability
        self.vocab_size = vocab_size
        self.train = train

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch_size = len(rows)
        input_ids = torch.full((batch_size, self.max_seq_len), self.special.pad, dtype=torch.long)
        time_gap_ids = torch.zeros((batch_size, self.max_seq_len), dtype=torch.long)
        session_flags = torch.zeros((batch_size, self.max_seq_len), dtype=torch.long)
        attention_mask = torch.zeros((batch_size, self.max_seq_len), dtype=torch.bool)
        next_labels = torch.empty(batch_size, dtype=torch.long)

        for row_idx, row in enumerate(rows):
            event_ids = [self.special.cls] + [int(value) for value in row["event_token_ids"]]
            gaps = [0] + [int(value) for value in row["time_gap_ids"]]
            sessions = [1] + [int(value) for value in row["session_flags"]]

            seq_len = min(len(event_ids), self.max_seq_len)
            input_ids[row_idx, :seq_len] = torch.tensor(event_ids[:seq_len], dtype=torch.long)
            time_gap_ids[row_idx, :seq_len] = torch.tensor(gaps[:seq_len], dtype=torch.long)
            session_flags[row_idx, :seq_len] = torch.tensor(sessions[:seq_len], dtype=torch.long).clamp(0, 1)
            attention_mask[row_idx, :seq_len] = True
            next_labels[row_idx] = int(row["next_event_token_id"])

        mlm_labels = torch.full_like(input_ids, -100)
        if self.train:
            input_ids, mlm_labels = self._mask_tokens(input_ids, attention_mask)

        return {
            "input_ids": input_ids,
            "time_gap_ids": time_gap_ids,
            "session_flags": session_flags,
            "attention_mask": attention_mask,
            "mlm_labels": mlm_labels,
            "next_labels": next_labels,
        }

    def _mask_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.vocab_size is None:
            raise ValueError("vocab_size is required for MLM masking")

        labels = input_ids.clone()
        special_mask = (
            (input_ids == self.special.pad)
            | (input_ids == self.special.cls)
            | (input_ids == self.special.mask)
        )
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        probability_matrix.masked_fill_(~attention_mask | special_mask, 0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Guarantee at least one masked event per sequence when possible.
        for row_idx in range(masked_indices.size(0)):
            if masked_indices[row_idx].any():
                continue
            candidate_positions = torch.where(attention_mask[row_idx] & ~special_mask[row_idx])[0]
            if len(candidate_positions) > 0:
                chosen = candidate_positions[torch.randint(len(candidate_positions), (1,))]
                masked_indices[row_idx, chosen] = True

        labels[~masked_indices] = -100

        replace_with_mask = torch.bernoulli(
            torch.full(labels.shape, self.mask_token_probability)
        ).bool() & masked_indices
        input_ids[replace_with_mask] = self.special.mask

        replace_with_random = torch.bernoulli(
            torch.full(labels.shape, self.random_token_probability)
        ).bool() & masked_indices & ~replace_with_mask
        random_tokens = torch.randint(4, self.vocab_size, labels.shape, dtype=torch.long)
        input_ids[replace_with_random] = random_tokens[replace_with_random]

        return input_ids, labels
