from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertMLP(nn.Module):
    """Two-layer expert MLP: D -> 2D -> D with GELU."""

    def __init__(self, hidden_size: int, dropout_prob: float = 0.0):
        super().__init__()
        inner_size = int(hidden_size) * 2
        self.fc1 = nn.Linear(hidden_size, inner_size)
        self.fc2 = nn.Linear(inner_size, hidden_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(max(float(dropout_prob), 0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class LatentMoETransition(nn.Module):
    """Latent transition used before each backbone latent step."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        use_shared_expert: bool = True,
        step_embed_max_steps: int = 32,
        expert_dropout: float = 0.0,
    ):
        super().__init__()
        hidden_size = int(hidden_size)
        num_experts = max(int(num_experts), 1)
        top_k = max(int(top_k), 1)
        top_k = min(top_k, num_experts)
        step_embed_max_steps = max(int(step_embed_max_steps), 1)
        expert_dropout = max(float(expert_dropout), 0.0)

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_shared_expert = bool(use_shared_expert)
        self.step_embed_max_steps = step_embed_max_steps
        self.expert_dropout = expert_dropout

        self.ln = nn.LayerNorm(hidden_size)
        self.step_embed = nn.Embedding(step_embed_max_steps, hidden_size)
        self.router = nn.Linear(hidden_size * 2, num_experts)
        self.experts = nn.ModuleList(
            [ExpertMLP(hidden_size, dropout_prob=expert_dropout) for _ in range(num_experts)]
        )
        self.shared_expert = (
            ExpertMLP(hidden_size, dropout_prob=expert_dropout) if self.use_shared_expert else None
        )

    def _build_step_ids(
        self, batch_size: int, step_id: Union[int, torch.Tensor], device: torch.device
    ) -> torch.Tensor:
        if torch.is_tensor(step_id):
            step_ids = step_id.to(device=device, dtype=torch.long).view(-1)
            if step_ids.numel() == 1 and batch_size > 1:
                step_ids = step_ids.expand(batch_size)
            if step_ids.numel() != batch_size:
                raise ValueError(
                    f"step_id shape mismatch: batch_size={batch_size}, got={tuple(step_ids.shape)}"
                )
        else:
            step_ids = torch.full((batch_size,), int(step_id), dtype=torch.long, device=device)
        return torch.clamp(step_ids, min=0, max=self.step_embed_max_steps - 1)

    def _normalize_context(
        self, z_t: torch.Tensor, context: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if context is None:
            return torch.zeros_like(z_t)
        context = context.to(device=z_t.device, dtype=z_t.dtype)
        if context.ndim == 1:
            context = context.unsqueeze(0)
        if context.shape[0] == 1 and z_t.shape[0] > 1:
            context = context.expand(z_t.shape[0], -1)
        if context.shape != z_t.shape:
            raise ValueError(
                f"context shape mismatch: expected {tuple(z_t.shape)}, got {tuple(context.shape)}"
            )
        return context

    def forward(
        self,
        z_t: torch.Tensor,
        step_id: Union[int, torch.Tensor],
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if z_t.ndim != 2:
            raise ValueError(f"z_t must be [B, D], got shape={tuple(z_t.shape)}")
        batch_size, hidden_size = z_t.shape
        if hidden_size != self.hidden_size:
            raise ValueError(
                f"hidden size mismatch: expected {self.hidden_size}, got {hidden_size}"
            )

        step_ids = self._build_step_ids(batch_size, step_id, z_t.device)
        step_emb = self.step_embed(step_ids).to(dtype=z_t.dtype)
        ctx = self._normalize_context(z_t, context)

        router_in = torch.cat([z_t + ctx, step_emb], dim=-1)
        gate_logits = self.router(router_in)
        gate_probs = F.softmax(gate_logits, dim=-1)

        topk_prob, topk_idx = torch.topk(gate_probs, k=self.top_k, dim=-1)
        u = self.ln(z_t)
        if u.is_floating_point() and u.dtype != z_t.dtype:
            u = u.to(dtype=z_t.dtype)

        if self.shared_expert is not None:
            mixed = self.shared_expert(u)
        else:
            mixed = torch.zeros_like(u)

        expert_mix = torch.zeros_like(u)
        for k_idx in range(self.top_k):
            idx_k = topk_idx[:, k_idx]
            prob_k = topk_prob[:, k_idx].unsqueeze(-1).to(dtype=u.dtype)
            out_k = torch.zeros_like(u)
            for expert_id, expert in enumerate(self.experts):
                mask = idx_k == expert_id
                if mask.any():
                    expert_out = expert(u[mask])
                    if expert_out.dtype != out_k.dtype:
                        expert_out = expert_out.to(dtype=out_k.dtype)
                    out_k[mask] = expert_out
            expert_mix = expert_mix + prob_k * out_k

        mixed = mixed + expert_mix
        if mixed.is_floating_point() and mixed.dtype != z_t.dtype:
            mixed = mixed.to(dtype=z_t.dtype)
        z_out = z_t + mixed

        avg_prob = gate_probs.mean(dim=0)
        target = torch.full_like(avg_prob, 1.0 / float(self.num_experts))
        balance_loss = ((avg_prob - target) ** 2).mean()
        router_entropy = -(gate_probs * gate_probs.clamp_min(1e-8).log()).sum(dim=-1).mean()

        aux = {
            "balance_loss": balance_loss,
            "router_entropy": router_entropy,
        }
        return z_out, aux
