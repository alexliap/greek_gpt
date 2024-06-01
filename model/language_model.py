import json
import os

import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import Adam
from mlx.utils import tree_flatten

from model.decoder import TransformerBlocks


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
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.context_len = context_len
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.n_experts = n_experts
        self.top_k = top_k
        self.dropout = dropout
        self.lr = lr

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

        self.optim = Adam(learning_rate=lr)

    def __call__(self, idxs):
        _, T = idxs.shape

        token_embed = self.embed_layer(idxs)
        position_embed = self.positional_embed(mx.arange(0, T))
        x = token_embed + position_embed
        x = self.blocks(x)
        x = self.layer_norm(x)
        logits = self.llm_head(x)
        logits = logits.reshape(-1, logits.shape[-1])

        return logits

    def inference(self, idxs):
        self.train(False)
        _, T = idxs.shape

        token_embed = self.embed_layer(idxs)
        position_embed = self.positional_embed(mx.arange(0, T))
        x = token_embed + position_embed
        x = self.blocks(x)
        x = self.layer_norm(x)
        logits = self.llm_head(x)
        logits = logits[:, -1, :]

        return logits

    def get_size(self):
        num_params = sum(v.size for _, v in tree_flatten(self.parameters()))
        return num_params

    def save_model(self, directory):
        if not os.path.exists(directory):
            os.mkdir(directory)

        model_config = {
            "vocab_size": self.vocab_size,
            "n_embed": self.n_embed,
            "context_len": self.context_len,
            "n_blocks": self.n_blocks,
            "n_heads": self.n_heads,
            "n_experts": self.n_experts,
            "top_k": self.top_k,
            "dropout": self.dropout,
            "lr": self.lr,
        }

        with open(directory + "model_config.json", "w") as outfile:
            json.dump(model_config, outfile)

        flat_params = tree_flatten(self.parameters())
        mx.save_safetensors(directory + "model_params", dict(flat_params))


class TransformerLLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embed: int,
        context_len: int,
        n_blocks: int,
        n_heads: int,
        hidden_multiplier: int = 2,
        dropout: float = 0.2,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.context_len = context_len
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_multiplier = hidden_multiplier
        self.dropout = dropout
        self.lr = lr

        self.embed_layer = nn.Embedding(vocab_size, n_embed)
        self.positional_embed = nn.Embedding(context_len, n_embed)
        self.blocks = TransformerBlocks(
            n_blocks=n_blocks,
            n_embed=n_embed,
            n_heads=n_heads,
            hidden_multiplier=hidden_multiplier,
            dropout=dropout,
        )
        self.layer_norm = nn.LayerNorm(n_embed)
        self.llm_head = nn.Linear(n_embed, vocab_size)

        self.optim = Adam(learning_rate=lr)

    def __call__(self, idxs):
        _, T = idxs.shape

        token_embed = self.embed_layer(idxs)
        position_embed = self.positional_embed(mx.arange(0, T))
        x = token_embed + position_embed
        x = self.blocks(x)
        x = self.layer_norm(x)
        logits = self.llm_head(x)
        logits = logits.reshape(-1, logits.shape[-1])

        return logits

    def inference(self, idxs):
        self.train(False)
        _, T = idxs.shape

        token_embed = self.embed_layer(idxs)
        position_embed = self.positional_embed(mx.arange(0, T))
        x = token_embed + position_embed
        x = self.blocks(x)
        x = self.layer_norm(x)
        logits = self.llm_head(x)
        logits = logits[:, -1, :]

        return logits

    def get_size(self):
        num_params = sum(v.size for _, v in tree_flatten(self.parameters()))
        return num_params

    def save_model(self, directory):
        if not os.path.exists(directory):
            os.mkdir(directory)

        model_config = {
            "vocab_size": self.vocab_size,
            "n_embed": self.n_embed,
            "context_len": self.context_len,
            "n_blocks": self.n_blocks,
            "n_heads": self.n_heads,
            "hidden_multiplier": self.hidden_multiplier,
            "dropout": self.dropout,
            "lr": self.lr,
        }

        with open(directory + "model_config.json", "w") as outfile:
            json.dump(model_config, outfile)

        flat_params = tree_flatten(self.parameters())
        mx.save_safetensors(directory + "model_params", dict(flat_params))


def load_model(directory, is_moe: bool):
    with open(directory + "model_config.json") as json_file:
        model_config = json.load(json_file)

    if is_moe:
        model = LanguageModel(**model_config)
    else:
        model = TransformerLLM(**model_config)

    model.load_weights(directory + "model_params.safetensors")

    return model
