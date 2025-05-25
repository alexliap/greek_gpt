import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionHead(nn.Module):
    """Single Self Attention Head."""

    def __init__(
        self, input_size: int, context_len: int, head_size: int, dropout: float = 0.2
    ):
        super().__init__()

        self.head_size = head_size

        self.query_proj = nn.Linear(
            in_features=input_size, out_features=head_size, bias=False
        )
        self.keys_proj = nn.Linear(
            in_features=input_size, out_features=head_size, bias=False
        )
        self.values_proj = nn.Linear(
            in_features=input_size, out_features=head_size, bias=False
        )

        self.register_buffer("tril", torch.tril(torch.ones(context_len, context_len)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, T, _ = x.shape

        keys = self.keys_proj(x)
        queries = self.query_proj(x)
        values = self.values_proj(x)

        W = torch.matmul(queries, torch.transpose(keys, -2, -1)) * self.head_size ** (
            -0.5
        )
        W = W.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        W = F.softmax(W, dim=-1)
        W = self.dropout(W)
        out = torch.matmul(W, values)

        return out


# B, T, C = 1, 8, 32
# head_size = 16
# head = SelfAttentionHead(C, head_size)
# x = torch.randn(B, T, C)
# print(head(x).shape)
