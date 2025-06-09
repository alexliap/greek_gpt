import argparse
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download folder from Azure")
    parser.add_argument("--vocab-size", required=True)
    args = parser.parse_args()

    vocab_size = int(args.vocab_size)

    tokenizer = AutoTokenizer.from_pretrained("tokenizer_" + vocab_size)

    dataset = load_dataset("wikimedia/wikipedia", "20231101.el")

    tokenized_data = tokenize_txt_dataset(tokenizer, dataset)

    flattened_data = [token for token_list in tokenized_data for token in token_list]

    val_data = [token for token_list in tokenized_data[::200] for token in token_list]

    with open(
        f"data/tokenized_data/train_tokenizer_{vocab_size}_data.pkl", "wb"
    ) as file:
        pickle.dump(flattened_data, file)

    with open(f"data/tokenized_data/val_tokenizer_{vocab_size}_data.pkl", "wb") as file:
        pickle.dump(val_data, file)
