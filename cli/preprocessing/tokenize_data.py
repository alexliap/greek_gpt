import os
import pickle

from datasets import load_dataset
from transformers import AutoTokenizer

from src.data_processing.txt_processing import tokenize_txt_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "true"


if not os.path.exists("data/"):
    os.mkdir("data/")
if not os.path.exists("data/tokenized_data/"):
    os.mkdir("data/tokenized_data/")

tokenizer_dict = {}
for vocab_size in ["1000", "2000", "5000", "10000"]:
    tokenizer_dict["tokenizer_" + vocab_size] = AutoTokenizer.from_pretrained(
        "tokenizer_" + vocab_size
    )

dataset = load_dataset("wikimedia/wikipedia", "20231101.el")

for tokenizer in tokenizer_dict.keys():
    tokenized_data = tokenize_txt_dataset(tokenizer_dict[tokenizer], dataset)

    flattened_data = [token for token_list in tokenized_data for token in token_list]

    val_data = [token for token_list in tokenized_data[::200] for token in token_list]

    with open(f"data/tokenized_data/train_{tokenizer}_data.pkl", "wb") as file:
        pickle.dump(flattened_data, file)

    with open(f"data/tokenized_data/val_{tokenizer}_data.pkl", "wb") as file:
        pickle.dump(val_data, file)
