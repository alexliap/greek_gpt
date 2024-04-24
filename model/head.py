import mlx.core as mx
import mlx.nn as nn
import numpy as np


class SelfAttentionHead(nn.Module):
    """Single Self Attention Head."""

    def __init__(self, input_size: int, head_size: int, dropout: float = 0.2):
        super().__init__()

        self.head_size = head_size

        self.query_proj = nn.Linear(
            input_dims=input_size, output_dims=head_size, bias=False
        )
        self.keys_proj = nn.Linear(
            input_dims=input_size, output_dims=head_size, bias=False
        )
        self.values_proj = nn.Linear(
            input_dims=input_size, output_dims=head_size, bias=False
        )

        self.dropout = nn.Dropout(dropout)

    def __call__(self, x):
        B, T, C = x.shape
        keys = self.keys_proj(x)
        queries = self.query_proj(x)
        values = self.values_proj(x)

        W = mx.matmul(keys, mx.transpose(queries, [0, 2, 1])) * self.head_size ** (-0.5)
        W = mx.tril(W, k=0)

        mask = np.tril(np.ones((B, T, T)))
        mask[mask == 0] = float("-inf")
        mask -= 1
        mask = mx.array(mask)
        W = nn.softmax(W + mask)
        W = self.dropout(W)
        out = mx.matmul(W, values)

        return out


# B, T, C = 1, 8, 32
# head_size = 16
# head = SelfAttentionHead(C, head_size)

# x = mx.random.uniform(0,5, (B, T, C))

# print(head(x).shape)
