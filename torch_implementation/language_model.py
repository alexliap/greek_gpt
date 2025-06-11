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
        dropout: float = 0.2,
    ):
        super().__init__()

        self.embed_layer = nn.Embedding(vocab_size + 257, n_embed)
        self.positional_embed = nn.Embedding(context_len, n_embed)
        self.blocks = TransformerBlocks(
            n_blocks=n_blocks,
            context_len=context_len,
            n_embed=n_embed,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.layer_norm = nn.LayerNorm(n_embed)
        self.llm_head = nn.Linear(n_embed, vocab_size + 257)

        self.context_len = context_len

    def forward(self, idxs):
        idxs = idxs.view(-1, idxs.size(0))
        _, T = idxs.shape

        token_embed = self.embed_layer(idxs)
        position_embed = self.positional_embed(torch.arange(0, T, device=idxs.device))
        x = token_embed + position_embed
        x = self.blocks(x)
        x = self.layer_norm(x)
        logits = self.llm_head(x)
        logits = logits.reshape(-1, logits.shape[-1])

        return logits

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class GreekGPTPretrain(L.LightningModule):
    def __init__(
        self,
        model: GreekGPT,
        lr: float = 3e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = model

        self.lr = lr

    def forward(self, idxs):
        return self.model(idxs)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch[0][0], batch[1][0]

        logits = self.forward(x)
        loss = F.cross_entropy(logits, y.view(-1))

        self.log("train_ce_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch[0][0], batch[1][0]

        logits = self.forward(x)
        loss = F.cross_entropy(logits, y.view(-1))

        self.log("val_ce_loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.lr)
        return optimizer

    def generate(
        self,
        query: str,
        tokenizer,
        max_tokens: int,
        temperature: float = 1.0,
    ):
        for _ in range(max_tokens):
            tokens = torch.tensor(tokenizer(query)["input_ids"]).reshape(1, -1)
            if len(tokens) > 256:
                tokens = tokens[-256:]
            out = self.forward(tokens) / temperature
            out = out[-1, :]
            s_out = F.softmax(out, dim=-1)

            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, _ = torch.topk(s_out, 50, dim=-1)

            chosen_token = torch.multinomial(topk_probs, 1)  # (B, 1)

            query += tokenizer.decode(chosen_token.item())

        return query
