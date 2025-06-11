import torch.nn as nn

from torch_implementation.experts import SparseMoE
from torch_implementation.multihead import MultiHeadAttention


class Transformer(nn.Module):
    def __init__(
        self,
        context_len: int,
        n_embed: int,
        n_heads: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.multi_attention = MultiHeadAttention(
            context_len=context_len, n_heads=n_heads, n_embed=n_embed
        )
        self.mlp = self.expert = nn.Sequential(
            nn.Linear(n_embed, 2 * n_embed),
            nn.ReLU(),
            nn.Linear(2 * n_embed, n_embed),
            nn.Dropout(dropout),
        )
        self.layernorm_1 = nn.LayerNorm(n_embed)
        self.layernorm_2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.multi_attention(self.layernorm_1(x))
        x = x + self.mlp(self.layernorm_2(x))
        return x


class TransformerBlocks(nn.Module):
    def __init__(
        self,
        n_blocks: int,
        context_len: int,
        n_embed: int,
        n_heads: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Transformer(
                    context_len=context_len,
                    n_embed=n_embed,
                    n_heads=n_heads,
                    dropout=dropout,
                )
                for _ in range(n_blocks)
            ]
        )
        self.laynernorm = nn.LayerNorm(n_embed)

    def forward(self, x):
        for transformer in self.blocks:
            x = transformer(x)
        x = self.laynernorm(x)
        return x


class MoETransformer(nn.Module):
    def __init__(
        self,
        context_len: int,
        n_embed: int,
        n_heads: int,
        n_experts: int,
        top_k: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.multi_attention = MultiHeadAttention(
            context_len=context_len, n_heads=n_heads, n_embed=n_embed
        )
        self.moe = SparseMoE(
            n_experts=n_experts, n_embed=n_embed, top_k=top_k, dropout=dropout
        )
        self.layernorm_1 = nn.LayerNorm(n_embed)
        self.layernorm_2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.multi_attention(self.layernorm_1(x))
        x = x + self.moe(self.layernorm_2(x))
        return x


class MoETransformerBlocks(nn.Module):
    def __init__(
        self,
        n_blocks: int,
        context_len: int,
        n_embed: int,
        n_heads: int,
        n_experts: int,
        top_k: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                MoETransformer(
                    context_len=context_len,
                    n_embed=n_embed,
                    n_heads=n_heads,
                    n_experts=n_experts,
                    top_k=top_k,
                    dropout=dropout,
                )
                for _ in range(n_blocks)
            ]
        )
        self.laynernorm = nn.LayerNorm(n_embed)

    def forward(self, x):
        for transformer in self.blocks:
            x = transformer(x)
        x = self.laynernorm(x)
        return x


# B, T, C = 1, 8, 32

# trans = Transformer(n_embed=C, n_heads=4)
# x = mx.random.randint(0,5, (B, T, C))
# print(trans(x).shape)

# blocks = TransformerBlocks(n_blocks=2, n_embed=C, n_heads=2)
# x = mx.random.randint(0,5, (B, T, C))
# print(blocks(x).shape)
