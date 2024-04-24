import os
import pickle

import nltk.data
from datasets import load_dataset
from tqdm import tqdm

if not os.path.exists("data/"):
    os.mkdir("data/")
if not os.path.exists("data/text_data/"):
    os.mkdir("data/text_data/")

sent_splitter = nltk.data.load("tokenizers/punkt/greek.pickle")
data = load_dataset("wikimedia/wikipedia", "20231101.el")


def add_special_tokens_to_data(dataset):
    data_list = []
    dataset = dataset["train"]
    for i in tqdm(range(len(dataset)), bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}"):
        text = "<SEP>".join(sent_splitter.tokenize(dataset[i]["text"]))
        text = "<BOS>" + text + "<EOS>"

        data_list.append(text)

    return data_list


data_list = add_special_tokens_to_data(data)

train_data = data_list[: int(len(data_list) * 0.8)]
test_data = data_list[int(len(data_list) * 0.8) :]

with open("data/text_data/train_text_dataset.pkl", "wb") as file:
    pickle.dump(train_data, file)
with open("data/text_data/test_text_dataset.pkl", "wb") as file:
    pickle.dump(test_data, file)

print("\nAdded tokens to the dataset (sep_token, eos_token, bos_token).")
print(f"Train dataset size: {len(train_data)}.")
print(f"Test dataset size: {len(test_data)}.")
