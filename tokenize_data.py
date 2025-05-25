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
    with open(f"data/tokenized_data/{tokenizer}_data.pkl", "wb") as file:
        pickle.dump(tokenized_data, file)

# for tokenizer in tokenizer_dict.keys():
#     test_tokenized_data = []
#     for text in tqdm(test_data, bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}"):
#         test_tokenized_data.append(tokenizer_dict[tokenizer](text)["input_ids"])
#     with open(f"data/tokenized_data/test_{tokenizer}_data.pkl", "wb") as file:
#         pickle.dump(test_tokenized_data, file)
