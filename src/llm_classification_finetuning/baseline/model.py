"""Frozen Qwen3 pairwise classifier and trainable comparison head."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from transformers import AutoModel


class PairwiseClassificationHead(nn.Module):
    """Classify explicit comparison features built from two pooled states."""

    def __init__(
        self, backbone_hidden_size: int, hidden_size: int, dropout: float
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(backbone_hidden_size * 4, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 3),
        )

    def forward(self, h_a: Tensor, h_b: Tensor) -> Tensor:
        """Return A-win, B-win, and tie logits for paired pooled states."""
        if h_a.shape != h_b.shape or h_a.ndim != 2:
            raise ValueError("h_a and h_b must have the same [batch, hidden] shape.")
        features = torch.cat((h_a, h_b, h_a - h_b, h_a * h_b), dim=-1)
        return self.network(features)


class PairwiseQwenClassifier(nn.Module):
    """Encode both branches with one frozen Qwen3 backbone and classify them."""

    def __init__(
        self,
        model_name: str,
        classifier_hidden_size: int,
        dropout: float,
        device: torch.device,
        dtype: torch.dtype,
        attention_implementation: str,
    ) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            model_name,
            dtype=dtype,
            attn_implementation=attention_implementation,
        )
        self.backbone.config.use_cache = False
        self.backbone.requires_grad_(False)
        self.backbone.to(device)
        self.backbone.eval()

        self.backbone_hidden_size = int(self.backbone.config.hidden_size)
        self.classifier = PairwiseClassificationHead(
            backbone_hidden_size=self.backbone_hidden_size,
            hidden_size=classifier_hidden_size,
            dropout=dropout,
        ).to(device)

    def train(self, mode: bool = True) -> PairwiseQwenClassifier:
        """Change head mode without ever enabling stochastic backbone behavior."""
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def encode(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Return last-token pooled states with shape [batch, 2, hidden]."""
        if input_ids.ndim != 3 or input_ids.shape[1] != 2:
            raise ValueError("input_ids must have shape [batch, 2, length].")
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids shape.")

        batch_size, branches, sequence_length = input_ids.shape
        flat_input_ids = input_ids.reshape(batch_size * branches, sequence_length)
        flat_attention_mask = attention_mask.reshape(
            batch_size * branches, sequence_length
        )
        outputs = self.backbone(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state.reshape(
            batch_size,
            branches,
            sequence_length,
            self.backbone_hidden_size,
        )

        positions = torch.arange(sequence_length, device=attention_mask.device)
        positions = positions.view(1, 1, sequence_length).expand_as(attention_mask)
        last_positions = positions.masked_fill(attention_mask.eq(0), -1).amax(dim=-1)
        if last_positions.lt(0).any():
            raise ValueError("Every response branch must contain at least one token.")

        gather_indices = last_positions[..., None, None].expand(
            -1, -1, 1, self.backbone_hidden_size
        )
        return hidden_states.gather(dim=2, index=gather_indices).squeeze(dim=2)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Return three pairwise logits from tokenized A/B inputs."""
        pooled = self.encode(input_ids, attention_mask)
        classifier_dtype = next(self.classifier.parameters()).dtype
        return self.classifier(
            pooled[:, 0].to(dtype=classifier_dtype),
            pooled[:, 1].to(dtype=classifier_dtype),
        )
