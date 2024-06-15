import torch
import torch.nn as nn

from torch_implementation.decoder import TransformerBlocks


class LanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embed: int,
        context_len: int,
        n_blocks: int,
        n_heads: int,
        n_experts: int = 4,
        top_k: int = 2,
        dropout: float = 0.2,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.embed_layer = nn.Embedding(vocab_size, n_embed)
        self.positional_embed = nn.Embedding(context_len, n_embed)
        self.blocks = TransformerBlocks(
            n_blocks=n_blocks,
            n_embed=n_embed,
            n_heads=n_heads,
            n_experts=n_experts,
            top_k=top_k,
            dropout=dropout,
        )
        self.layer_norm = nn.LayerNorm(n_embed)
        self.llm_head = nn.Linear(n_embed, vocab_size)

        self.optim = torch.optim.Adam(params=self.parameters(), lr=lr)

    def forward(self, idxs):
        _, T = idxs.shape

        token_embed = self.embed_layer(idxs)
        position_embed = self.positional_embed(torch.arange(0, T))
        x = token_embed + position_embed
        x = self.blocks(x)
        x = self.layer_norm(x)
        logits = self.llm_head(x)
        logits = logits.reshape(-1, logits.shape[-1])

        return logits

    # def get_size(self):
    #     num_params = sum(v.size for _, v in tree_flatten(self.parameters()))
    #     return num_params


# B, T = 128, 256
# model = LanguageModel(
#     vocab_size=5000,
#     n_embed=256,
#     context_len=T,
#     n_blocks=8,
#     n_heads=8,
#     n_experts=4,
#     top_k=2,
#     lr=1e-4,
# ).bfloat16()
# total_params = sum(p.numel() for p in model.parameters())
# print(round(total_params/1e6, 3))
# x = torch.randint(0, 5000, (B, T))
# print(model(x).shape)
