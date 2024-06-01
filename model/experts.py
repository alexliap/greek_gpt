import mlx.core as mx
import mlx.nn as nn
import numpy as np

from model.router import Router


class Expert(nn.Module):
    def __init__(self, n_embed: int, hidden_multiplier: int = 2, dropout: float = 0.2):
        super().__init__()

        self.expert = nn.Sequential(
            nn.Linear(n_embed, hidden_multiplier * n_embed),
            nn.ReLU(),
            nn.Linear(hidden_multiplier * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def __call__(self, x):
        x = self.expert(x)
        return x


class SparseMoE(nn.Module):
    def __init__(
        self,
        n_experts: int,
        n_embed: int,
        hidden_multiplier: int = 2,
        top_k: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.router = Router(n_embed=n_embed, n_experts=n_experts, top_k=top_k)
        self.experts = [
            Expert(
                n_embed=n_embed, hidden_multiplier=hidden_multiplier, dropout=dropout
            )
            for _ in range(n_experts)
        ]

    def __call__(self, x):
        B, T, C = x.shape
        gating_output, indices = self.router(x)

        final_output = mx.zeros_like(x)
        final_output = final_output.reshape(-1, final_output.shape[-1])

        flat_x = x.reshape((-1, x.shape[-1]))
        flat_gating_output = gating_output.reshape((-1, gating_output.shape[-1]))

        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(
                axis=-1
            )  # which batches pass from which experts
            flat_mask = expert_mask.reshape(-1)
            flat_mask = mx.array(np.where(flat_mask)[0])
            if flat_mask.any():
                expert_input = flat_x[flat_mask]
                expert_output = expert(expert_input)

                # Extract and apply gating scores
                gating_scores = flat_gating_output[flat_mask, i]
                weighted_output = expert_output * gating_scores.reshape(
                    gating_scores.size, 1
                )

                final_output[flat_mask] += weighted_output

        final_output = final_output.reshape(B, T, C)

        return final_output
