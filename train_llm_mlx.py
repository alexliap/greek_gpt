import pickle

import mlx.core as mx
import mlx.nn as nn
from tqdm import trange
from transformers import AutoTokenizer

from core.training import get_batch, loss_fn, training_step, validation_step
from model.language_model import load_model

tokenizer = "tokenizer_10000"

with open(f"data/tokenized_data/train_{tokenizer}_data.pkl", "rb") as file:
    train_data = pickle.load(file)
with open(f"data/tokenized_data/test_{tokenizer}_data.pkl", "rb") as file:
    test_data = pickle.load(file)

train_data = [token for sents in train_data for token in sents]
test_data = [token for sents in test_data for token in sents]

train_data = mx.array(train_data)
test_data = mx.array(test_data)

epochs = 8000
eval_interval = 50
save_interval = 200
block_size = 256
batch_size = 128

tokenizer = AutoTokenizer.from_pretrained(tokenizer)
vocab_size = len(tokenizer.get_vocab())
# model = LanguageModel(
#     vocab_size=vocab_size,
#     n_embed=256,
#     context_len=256,
#     n_blocks=6,
#     n_heads=4,
#     n_experts=4,
#     top_k=2,
#     dropout=0.2,
#     lr=3e-4
# )
model = load_model("trained_models/moe_model_10k_3200_4.219_4.025/", is_moe=True)
pretrained_epochs = 3200

model_size = model.get_size()

print(f"Transformer Model Size: {model_size} parameters.")

# training loop
loss_and_grad = nn.value_and_grad(model, loss_fn)
bar = trange(
    pretrained_epochs + 1,
    epochs + 1,
    leave=True,
    bar_format="{l_bar}{bar:15}{r_bar}{bar:-15b}",
)

val_str = ""
for i in bar:
    x, y = get_batch(train_data, batch_size, block_size)
    train_loss = training_step(model, loss_and_grad, x, y)

    train_str = f"Epoch {i} Train Loss: {round(train_loss.item(), 3)}"

    if i % eval_interval == 0:
        x_val, y_val = get_batch(test_data, 3 * batch_size, block_size)
        val_loss = validation_step(model, x_val, y_val)
        val_str = f"Epoch {i} Validation Loss: {round(val_loss.item(), 3)}"
    report_str = train_str + " | " + val_str
    bar.set_description(report_str)
    bar.refresh()

    if i % save_interval == 0:
        model.save_model(
            f"trained_models/moe_model_10k_{i}_{round(train_loss.item(), 3)}_{round(val_loss.item(), 3)}/"
        )
