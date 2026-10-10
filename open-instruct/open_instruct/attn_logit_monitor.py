"""Track the maximum pre-softmax attention logit during training, for logging to wandb.

FlashAttention/SDPA never materialize QK^T, so we wrap the model's entry in transformers'
``ALL_ATTENTION_FUNCTIONS``. The wrapper is a passthrough, except inside ``track(batch)``,
where it also computes ``(Q @ K^T) * scaling`` from the post-RoPE query/key states it receives
and folds its max into a running per-layer max. Only causal (query, key) pairs within the same
sequence count: padding and other documents in a packed batch are masked out. Logits are
computed in query chunks so the full (heads, S, S) matrix is never held at once.
"""

from contextlib import contextmanager

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

QUERY_CHUNK_SIZE = 256


def _segment_ids(batch: dict, seq_len: int, device: torch.device) -> torch.Tensor:
    """(B, S) sequence id per token; -1 marks padding. Tokens attend only within their id."""
    positions = torch.arange(seq_len, device=device)
    if "cu_seq_lens_q" in batch:
        cu_seq_lens = batch["cu_seq_lens_q"].to(device)
        seg = torch.searchsorted(cu_seq_lens[1:], positions, right=True)
        return torch.where(positions < cu_seq_lens[-1], seg, -1)[None]
    if "attention_mask" in batch:
        return torch.where(batch["attention_mask"].to(device).bool(), 0, -1)
    if "position_ids" in batch:
        return (batch["position_ids"].to(device) == 0).long().cumsum(-1) - 1
    raise ValueError("Expected cu_seq_lens_q, attention_mask, or position_ids in batch.")


class AttentionLogitMonitor:
    def __init__(self, attn_implementation: str, num_layers: int, device: torch.device):
        self.layer_max = torch.full((num_layers,), float("-inf"), device=device)
        self._batch = None
        original = ALL_ATTENTION_FUNCTIONS.get_interface(attn_implementation, None)
        if original is None:
            raise ValueError(
                f"Cannot track attention logits with attn_implementation={attn_implementation!r}: "
                "it is not dispatched through ALL_ATTENTION_FUNCTIONS. Use flash_attention_2 or sdpa."
            )

        def wrapped(module, query, key, value, attention_mask, *args, **kwargs):
            if self._batch is not None:
                self._record(module.layer_idx, query, key, kwargs.get("scaling"))
            return original(module, query, key, value, attention_mask, *args, **kwargs)

        ALL_ATTENTION_FUNCTIONS[attn_implementation] = wrapped

    @torch.no_grad()
    def _record(self, layer_idx: int, query: torch.Tensor, key: torch.Tensor, scaling: float | None):
        # query: (B, H, S, D); key: (B, H_kv, S, D), not yet expanded for GQA.
        query, key = query.detach(), key.detach()
        key = key.repeat_interleave(query.shape[1] // key.shape[1], dim=1)
        if scaling is None:
            scaling = query.shape[-1] ** -0.5
        seq_len = query.shape[2]
        seg = _segment_ids(self._batch, seq_len, query.device)  # (B, S)
        key_pos = torch.arange(seq_len, device=query.device)
        layer_max = self.layer_max[layer_idx]
        for start in range(0, seq_len, QUERY_CHUNK_SIZE):
            end = min(start + QUERY_CHUNK_SIZE, seq_len)
            logits = torch.matmul(query[:, :, start:end], key.transpose(-1, -2)).float() * scaling
            q_seg = seg[:, start:end, None]
            allowed = (q_seg == seg[:, None, :]) & (q_seg >= 0)
            allowed &= key_pos[None, None, :] <= torch.arange(start, end, device=query.device)[None, :, None]
            logits.masked_fill_(~allowed[:, None], float("-inf"))
            layer_max = torch.maximum(layer_max, logits.amax())
        self.layer_max[layer_idx] = layer_max

    @contextmanager
    def track(self, batch: dict):
        """Fold this forward pass's logits into the running max."""
        self._batch = batch
        try:
            yield
        finally:
            self._batch = None

    def pop(self) -> torch.Tensor:
        """Return the per-layer max since the last pop, and reset it."""
        layer_max = self.layer_max.clone()
        self.layer_max.fill_(float("-inf"))
        return layer_max
