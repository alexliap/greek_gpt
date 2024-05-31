import json
import os

import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import Adam
from mlx.utils import tree_flatten


class NGram(nn.Module):
    def __init__(
        self,
        context_len: int,
        n_embed: int,
        vocab_size: int,
        lr: float,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.context_len = context_len
        self.lr = lr

        self.n_gram = nn.Sequential(nn.Embedding(vocab_size, n_embed))
        self.head = nn.Sequential(
            nn.Linear(n_embed * context_len, n_embed * context_len),
            nn.Dropout(dropout),
            nn.Linear(n_embed * context_len, vocab_size),
        )

        self.optim = Adam(learning_rate=lr)

    def __call__(self, idxs):
        B, _ = idxs.shape[0], idxs.shape[1]

        out = self.n_gram(idxs)
        out = self.head(out.reshape(B, -1))

        return out

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
            "lr": self.lr,
        }

        with open(directory + "model_config.json", "w") as outfile:
            json.dump(model_config, outfile)

        flat_params = tree_flatten(self.parameters())
        mx.save_safetensors(directory + "model_params", dict(flat_params))


def load_ngram(directory):
    with open(directory + "model_config.json") as json_file:
        model_config = json.load(json_file)

    model = NGram(
        vocab_size=model_config["vocab_size"],
        n_embed=model_config["n_embed"],
        context_len=model_config["context_len"],
        lr=model_config["lr"],
    )
    model.load_weights(directory + "model_params.safetensors")

    return model
