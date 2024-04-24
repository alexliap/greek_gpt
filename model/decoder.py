import mlx.nn as nn

from model.experts import SparseMoE
from model.multihead import MultiHeadAttention


class Transformer(nn.Module):
    def __init__(
        self,
        n_embed: int,
        n_heads: int,
        n_experts: int,
        top_k: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.multi_attention = MultiHeadAttention(n_heads=n_heads, n_embed=n_embed)
        # TODO: to be replaces to MoE
        self.moe = SparseMoE(
            n_experts=n_experts, n_embed=n_embed, top_k=top_k, dropout=dropout
        )
        self.layernorm_1 = nn.LayerNorm(n_embed)
        self.layernorm_2 = nn.LayerNorm(n_embed)

    def __call__(self, x):
        x = x + self.multi_attention(self.layernorm_1(x))
        x = x + self.moe(self.layernorm_2(x))
        return x


class TransformerBlocks(nn.Module):
    def __init__(
        self,
        n_blocks: int,
        n_embed: int,
        n_heads: int,
        n_experts: int,
        top_k: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.blocks = [
            Transformer(
                n_embed=n_embed,
                n_heads=n_heads,
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


# B, T, C = 1, 8, 32

# trans = Transformer(n_embed=C, n_heads=4)

# x = mx.random.randint(0,5, (B, T, C))

# print(trans(x).shape)

# blocks = TransformerBlocks(n_blocks=2, n_embed=C, n_heads=2)

# x = mx.random.randint(0,5, (B, T, C))

# print(blocks(x).shape)
