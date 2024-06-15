import torch
import torch.nn as nn
from model_pytorch.router import Router


class Expert(nn.Module):
    def __init__(self, n_embed: int, dropout: float = 0.2):
        super().__init__()

        self.expert = nn.Sequential(
            nn.Linear(n_embed, 2 * n_embed),
            nn.ReLU(),
            nn.Linear(2 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.expert(x)
        return x


class SparseMoE(nn.Module):
    def __init__(
        self, n_experts: int, n_embed: int, top_k: int = 2, dropout: float = 0.2
    ):
        super().__init__()

        self.router = Router(n_embed=n_embed, n_experts=n_experts, top_k=top_k)
        self.experts = nn.ModuleList(
            [Expert(n_embed=n_embed, dropout=dropout) for _ in range(n_experts)]
        )

    def forward(self, x):
        gating_output, indices = self.router(x)
        final_output = torch.zeros_like(x)

        # Reshape inputs for batch processing
        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        # Process each expert in parallel
        for i, expert in enumerate(self.experts):
            # Create a mask for the inputs where the current expert is in top-k
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)

            if flat_mask.any():
                expert_input = flat_x[flat_mask]
                expert_output = expert(expert_input)

                # Extract and apply gating scores
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores

                # Update final output additively by indexing and adding
                final_output[expert_mask] += weighted_output.squeeze(1)

        return final_output
