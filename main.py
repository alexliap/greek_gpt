import warnings
warnings.filterwarnings("ignore")

import os
import pickle
from tqdm import trange
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from transformers import AutoTokenizer
from model.language_model import LanguageModel, load_model
from model.training import training_step, validation_step, loss_fn, get_batch

tokenizer = "tokenizer_5000"

with open(f"data/tokenized_data/train_{tokenizer}_data.pkl", 'rb') as file:
    train_data = pickle.load(file)
with open(f"data/tokenized_data/test_{tokenizer}_data.pkl", 'rb') as file:
    test_data = pickle.load(file)

train_data = [token for sents in train_data for token in sents]
test_data = [token for sents in test_data for token in sents]

epochs = 8000
eval_interval = 50
save_interval = 100
block_size = 256
batch_size = 128

tokenizer = AutoTokenizer.from_pretrained(tokenizer)
vocab_size = len(tokenizer.get_vocab())
model = load_model("trained_models/model_5500_3.766_3.668/")
pretrained_epochs = 5500

model_size = model.get_size()

print(f"MoE Transformer Model Size: {model_size} parameters.")

# training loop
loss_and_grad = nn.value_and_grad(model, loss_fn)
bar = trange(pretrained_epochs+1, epochs+1, leave=True, bar_format='{l_bar}{bar:15}{r_bar}{bar:-15b}')
val_str = ""
for i in bar:
    x, y = get_batch(train_data, batch_size, block_size, tokenizer)
    train_loss = training_step(model, loss_and_grad, x, y)

    train_str = f"Epoch {i} Train Loss: {round(train_loss.item(), 3)}"

    if i % eval_interval == 0:
        x_val, y_val = get_batch(test_data, 6*batch_size, block_size, tokenizer)
        val_loss = validation_step(model, x_val, y_val)
        val_str = f"Epoch {i} Validation Loss: {round(val_loss.item(), 3)}"
    report_str = train_str + " | " + val_str
    bar.set_description(report_str)
    bar.refresh()

    if i % save_interval == 0:
        model.save_model(f"trained_models/model_{i}_{round(train_loss.item(), 3)}_{round(val_loss.item(), 3)}/")
