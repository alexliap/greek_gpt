import os
import pickle

from tqdm import tqdm
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "true"

with open("data/text_data/train_text_dataset.pkl", "rb") as file:
    train_data = pickle.load(file)
with open("data/text_data/test_text_dataset.pkl", "rb") as file:
    test_data = pickle.load(file)

if not os.path.exists("data/"):
    os.mkdir("data/")
if not os.path.exists("data/tokenized_data/"):
    os.mkdir("data/tokenized_data/")

tokenizer_dict = {}
for item in ["1000", "2000", "5000", "10000"]:
    tokenizer_dict["tokenizer_" + item] = AutoTokenizer.from_pretrained(
        "tokenizer_" + item
    )

for tokenizer in tokenizer_dict.keys():
    train_tokenized_data = []
    for text in tqdm(train_data, bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}"):
        train_tokenized_data.append(tokenizer_dict[tokenizer](text)["input_ids"])
    with open(f"data/tokenized_data/train_{tokenizer}_data.pkl", "wb") as file:
        pickle.dump(train_tokenized_data, file)

for tokenizer in tokenizer_dict.keys():
    test_tokenized_data = []
    for text in tqdm(test_data, bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}"):
        test_tokenized_data.append(tokenizer_dict[tokenizer](text)["input_ids"])
    with open(f"data/tokenized_data/test_{tokenizer}_data.pkl", "wb") as file:
        pickle.dump(test_tokenized_data, file)
