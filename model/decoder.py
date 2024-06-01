from typing import Union

import mlx.nn as nn

from model.experts import Expert, SparseMoE
from model.multihead import MultiHeadAttention


class Transformer(nn.Module):
    def __init__(
        self,
        n_embed: int,
        n_heads: int,
        hidden_multiplier: int = 2,
        n_experts: Union[int, None] = None,
        top_k: Union[int, None] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.multi_attention = MultiHeadAttention(n_heads=n_heads, n_embed=n_embed)

        if n_experts is not None and top_k is not None:
            self.moe = SparseMoE(
                n_experts=n_experts,
                n_embed=n_embed,
                hidden_multiplier=hidden_multiplier,
                top_k=top_k,
                dropout=dropout,
            )
            self.is_moe = True
        else:
            self.mlp = Expert(
                n_embed=n_embed, hidden_multiplier=hidden_multiplier, dropout=dropout
            )
            self.is_moe = False

        self.layernorm_1 = nn.LayerNorm(n_embed)
        self.layernorm_2 = nn.LayerNorm(n_embed)

    def __call__(self, x):
        x = x + self.multi_attention(self.layernorm_1(x))
        if self.is_moe:
            x = x + self.moe(self.layernorm_2(x))
        else:
            x = x + self.mlp(self.layernorm_2(x))
        return x


class TransformerBlocks(nn.Module):
    def __init__(
        self,
        n_blocks: int,
        n_embed: int,
        n_heads: int,
        hidden_multiplier: int = 2,
        n_experts: Union[int, None] = None,
        top_k: Union[int, None] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.blocks = [
            Transformer(
                n_embed=n_embed,
                n_heads=n_heads,
                hidden_multiplier=hidden_multiplier,
                n_experts=n_experts,
                top_k=top_k,
                dropout=dropout,
            )
            for _ in range(n_blocks)
        ]
        self.laynernorm = nn.LayerNorm(n_embed)

    def __call__(self, x):
        for transformer in self.blocks:
            x = transformer(x)
        x = self.laynernorm(x)
        return x
