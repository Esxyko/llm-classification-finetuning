"""Frozen Qwen3 mono-input encoder and trainable classification head."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from transformers import AutoModel


class MonoInputClassificationHead(nn.Module):
    """Classify one pooled comparison state as A-win, B-win, or tie."""

    def __init__(
        self,
        backbone_hidden_size: int,
        hidden_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(backbone_hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 3),
        )

    def forward(self, features: Tensor) -> Tensor:
        """Return three logits for a batch of pooled comparison states."""
        if features.ndim != 2:
            raise ValueError("features must have shape [batch, hidden].")
        return self.network(features)


class FrozenQwenEncoder(nn.Module):
    """Encode mono-input texts across frozen Qwen replicas."""

    def __init__(
        self,
        model_name: str,
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

    def train(self, mode: bool = True) -> FrozenQwenEncoder:
        """Keep every frozen backbone in evaluation mode."""
        super().train(False)
        self.backbones.eval()
        return self

    @torch.no_grad()
    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Return final non-padding-token states with shape [batch, hidden]."""
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, length].")
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids shape.")

        input_chunks = input_ids.tensor_split(len(self.backbones), dim=0)
        mask_chunks = attention_mask.tensor_split(len(self.backbones), dim=0)
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
        if not pooled_chunks:
            raise ValueError("The encoder received an empty batch.")
        return torch.cat(pooled_chunks, dim=0)

    @staticmethod
    def _last_token_pool(hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        sequence_length = attention_mask.shape[1]
        positions = torch.arange(sequence_length, device=attention_mask.device)
        positions = positions.view(1, sequence_length).expand_as(attention_mask)
        last_positions = positions.masked_fill(attention_mask.eq(0), -1).amax(dim=-1)
        if last_positions.lt(0).any():
            raise ValueError("Every mono input must contain at least one token.")
        gather_indices = last_positions[:, None, None].expand(
            -1,
            1,
            hidden_states.shape[-1],
        )
        return hidden_states.gather(dim=1, index=gather_indices).squeeze(dim=1)


__all__ = (
    "FrozenQwenEncoder",
    "MonoInputClassificationHead",
)
