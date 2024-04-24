import os
import pickle
from typing import List

from tqdm import tqdm
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "true"

with open("data/text_data/train_text_dataset.pkl", "rb") as file:
    data = pickle.load(file)

print(f"\n Number of text samples in dataset: {len(data)} \n")

tokenizer = AutoTokenizer.from_pretrained(
    "gpt2",
    trust_remote_code=True,
    bos_token="<BOS>",
    eos_token="<EOS>",
    unk_token="<UNK>",
    sep_token="<SEP>",
    pad_token="<PAD>",
    split_special_tokens=False,
    model_max_length=int(3e5),
)


def batch_iterator(dataset: List[str], batch_size: int):
    for i in tqdm(
        range(0, len(dataset), batch_size),
        bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
    ):
        yield dataset[i : i + batch_size]


tokenizer_1000 = tokenizer.train_new_from_iterator(
    batch_iterator(data, 1000), vocab_size=1000
)
tokenizer_2000 = tokenizer.train_new_from_iterator(
    batch_iterator(data, 1000), vocab_size=2000
)
tokenizer_5000 = tokenizer.train_new_from_iterator(
    batch_iterator(data, 1000), vocab_size=5000
)
tokenizer_10000 = tokenizer.train_new_from_iterator(
    batch_iterator(data, 1000), vocab_size=10_000
)

tokenizer_1000.save_pretrained("tokenizer_1000/")
tokenizer_2000.save_pretrained("tokenizer_2000/")
tokenizer_5000.save_pretrained("tokenizer_5000/")
tokenizer_10000.save_pretrained("tokenizer_10000/")
