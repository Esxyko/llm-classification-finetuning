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
    """Encode paired branches across frozen Qwen3 replicas and classify them."""

    def __init__(
        self,
        model_name: str,
        classifier_hidden_size: int,
        dropout: float,
        devices: tuple[torch.device, ...],
        dtype: torch.dtype,
        attention_implementation: str,
    ) -> None:
        super().__init__()
        if not devices:
            raise ValueError("At least one CUDA device is required.")
        self.devices = devices
        self.primary_device = devices[0]
        self.backbones = nn.ModuleList()
        backbone_hidden_size: int | None = None
        for device in devices:
            backbone = AutoModel.from_pretrained(
                model_name,
                dtype=dtype,
                attn_implementation=attention_implementation,
            )
            backbone.config.use_cache = False
            backbone.requires_grad_(False)
            backbone.to(device)
            backbone.eval()
            current_hidden_size = int(backbone.config.hidden_size)
            if (
                backbone_hidden_size is not None
                and current_hidden_size != backbone_hidden_size
            ):
                raise ValueError("Backbone replicas have inconsistent hidden sizes.")
            backbone_hidden_size = current_hidden_size
            self.backbones.append(backbone)

        if backbone_hidden_size is None:
            raise ValueError("Could not initialize a Qwen backbone replica.")
        self.backbone_hidden_size = backbone_hidden_size
        self.classifier = PairwiseClassificationHead(
            backbone_hidden_size=self.backbone_hidden_size,
            hidden_size=classifier_hidden_size,
            dropout=dropout,
        ).to(self.primary_device)

    def train(self, mode: bool = True) -> PairwiseQwenClassifier:
        """Change head mode without ever enabling stochastic backbone behavior."""
        super().train(mode)
        self.backbones.eval()
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
        input_chunks = flat_input_ids.tensor_split(len(self.backbones), dim=0)
        mask_chunks = flat_attention_mask.tensor_split(len(self.backbones), dim=0)
        pooled_chunks: list[Tensor] = []
        for backbone, device, ids, mask in zip(
            self.backbones,
            self.devices,
            input_chunks,
            mask_chunks,
            strict=True,
        ):
            if ids.shape[0] == 0:
                continue
            device_ids = ids.to(device, non_blocking=True)
            device_mask = mask.to(device, non_blocking=True)
            hidden_states = backbone(
                input_ids=device_ids,
                attention_mask=device_mask,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state
            pooled_chunks.append(
                self._last_token_pool(hidden_states, device_mask).to(
                    self.primary_device,
                    non_blocking=True,
                )
            )

        pooled = torch.cat(pooled_chunks, dim=0)
        return pooled.reshape(batch_size, branches, self.backbone_hidden_size)

    @staticmethod
    def _last_token_pool(hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """Pool the final non-padding token before transferring across devices."""
        sequence_length = attention_mask.shape[1]
        positions = torch.arange(sequence_length, device=attention_mask.device)
        positions = positions.view(1, sequence_length).expand_as(attention_mask)
        last_positions = positions.masked_fill(attention_mask.eq(0), -1).amax(dim=-1)
        if last_positions.lt(0).any():
            raise ValueError("Every response branch must contain at least one token.")
        gather_indices = last_positions[:, None, None].expand(
            -1,
            1,
            hidden_states.shape[-1],
        )
        return hidden_states.gather(dim=1, index=gather_indices).squeeze(dim=1)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Return three pairwise logits from tokenized A/B inputs."""
        pooled = self.encode(input_ids, attention_mask)
        classifier_dtype = next(self.classifier.parameters()).dtype
        return self.classifier(
            pooled[:, 0].to(dtype=classifier_dtype),
            pooled[:, 1].to(dtype=classifier_dtype),
        )
