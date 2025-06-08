import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_implementation.decoder import TransformerBlocks


class GreekGPT(nn.Module):
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
    ):
        super().__init__()

        self.embed_layer = nn.Embedding(vocab_size, n_embed)
        self.positional_embed = nn.Embedding(context_len, n_embed)
        self.blocks = TransformerBlocks(
            n_blocks=n_blocks,
            context_len=context_len,
            n_embed=n_embed,
            n_heads=n_heads,
            n_experts=n_experts,
            top_k=top_k,
            dropout=dropout,
        )
        self.layer_norm = nn.LayerNorm(n_embed)
        self.llm_head = nn.Linear(n_embed, vocab_size)

        self.context_len = context_len

    def forward(self, idxs):
        # idxs = idxs.view(-1, self.context_len)
        _, T = idxs.shape

        token_embed = self.embed_layer(idxs)
        position_embed = self.positional_embed(torch.arange(0, T))
        x = token_embed + position_embed
        x = self.blocks(x)
        x = self.layer_norm(x)
        logits = self.llm_head(x)
        logits = logits.reshape(-1, logits.shape[-1])

        return logits

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class LanguageModel(L.LightningModule):
    def __init__(
        self,
        model: GreekGPT,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model

        self.lr = lr

    def forward(self, idxs):
        return self.model(idxs)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch

        logits = self.forward(x)
        loss = F.cross_entropy(logits, y.view(-1), reduction="mean")

        self.log("train_ce_loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.lr)
        return optimizer
