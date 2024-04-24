import mlx.core as mx
import mlx.nn as nn

from model.head import SelfAttentionHead


class MultiHeadAttention(nn.Module):
    """Multihead Attention module."""

    def __init__(self, n_heads: int, n_embed: int, dropout: float = 0.2):
        super().__init__()
        self.n_head = n_heads
        self.n_embed = n_embed
        self.head_size = n_embed // n_heads
        self.heads = [
            SelfAttentionHead(input_size=self.n_embed, head_size=self.head_size)
            for _ in range(self.n_head)
        ]
        self.projection = nn.Linear(self.n_head * self.head_size, self.n_embed)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x):
        out = mx.concatenate([head(x) for head in self.heads], axis=-1)
        out = self.projection(out)
        out = self.dropout(out)
        return out


# B, T, C = 1, 8, 32

# heads = MultiHeadAttention(4, C)

# x = mx.random.uniform(0,5, (B, T, C))

# print(heads(x).shape)
