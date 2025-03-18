import pickle

import mlx.core as mx
import mlx.nn as nn
from tqdm import trange
from transformers import AutoTokenizer

from benchmarks.n_grams import NGram
from core.training import get_batch, loss_fn, training_step, validation_step

tokenizer = "tokenizer_5000"
tokenizer_obj = AutoTokenizer.from_pretrained(tokenizer)

epochs = 8000
eval_interval = 50
save_interval = 500
# n-gram: n = context_len + 1
context_len = 4
batch_size = 1000

vocab_size = len(tokenizer_obj.get_vocab())
model = NGram(context_len=context_len, n_embed=256, vocab_size=vocab_size, lr=1e-4)

# minus 1 beacase BOS and EOS token already exist one time per piece of text
additional_bos = (context_len - 1) * [tokenizer_obj.bos_token_id]
additional_eos = (context_len - 1) * [tokenizer_obj.eos_token_id]

with open(f"data/tokenized_data/train_{tokenizer}_data.pkl", "rb") as file:
    train_data = pickle.load(file)
    train_data = [additional_bos + text + additional_eos for text in train_data]
with open(f"data/tokenized_data/test_{tokenizer}_data.pkl", "rb") as file:
    test_data = pickle.load(file)
    test_data = [additional_bos + text + additional_eos for text in test_data]

train_data = [token for sents in train_data for token in sents]
test_data = [token for sents in test_data for token in sents]

train_data = mx.array(train_data)
test_data = mx.array(test_data)

pretrained_epochs = 0

model_size = model.get_size()

print(f"N-Gram Model Size: {model_size} parameters.")

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
    x, y = get_batch(train_data, batch_size, context_len)
    y = y[:, context_len - 1]
    train_loss = training_step(model, loss_and_grad, x, y)

    train_str = f"Epoch {i} Train Loss: {round(train_loss.item(), 3)}"

    if i % eval_interval == 0:
        x_val, y_val = get_batch(test_data, 6 * batch_size, context_len)
        y_val = y_val[:, context_len - 1]
        val_loss = validation_step(model, x_val, y_val)
        val_str = f"Epoch {i} Validation Loss: {round(val_loss.item(), 3)}"
    report_str = train_str + " | " + val_str
    bar.set_description(report_str)
    bar.refresh()

    if i % save_interval == 0:
        model.save_model(
            f"benchmarks/challenger_models/{context_len + 1}-gram_{i}_\
{round(train_loss.item(), 3)}_{round(val_loss.item(), 3)}/"
        )
