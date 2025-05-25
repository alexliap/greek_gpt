import os

from tqdm import tqdm
from transformers import AutoTokenizer

UNICODE_CHARS_SIZE = 257

os.environ["TOKENIZERS_PARALLELISM"] = "true"

with open("data/text_data/txt_dataset.txt", "r", encoding="utf-8") as file:
    dataset = file.read()

tokenizer = AutoTokenizer.from_pretrained(
    "gpt2",
    trust_remote_code=True,
    # bos_token="<BOS>",
    # eos_token="<EOS>",
    # unk_token="<UNK>",
    # sep_token="<SEP>",
    # pad_token="<PAD>",
    split_special_tokens=False,
    model_max_length=int(3e5),
)


def batch_iterator(dataset: str, batch_size: int):
    for i in tqdm(
        range(0, len(dataset), batch_size),
        bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
    ):
        yield dataset[i : i + batch_size]


tokenizer_1000 = tokenizer.train_new_from_iterator(
    batch_iterator(dataset, 10_000), vocab_size=1000 + UNICODE_CHARS_SIZE
)
tokenizer_2000 = tokenizer.train_new_from_iterator(
    batch_iterator(dataset, 10_000), vocab_size=2000 + UNICODE_CHARS_SIZE
)
tokenizer_5000 = tokenizer.train_new_from_iterator(
    batch_iterator(dataset, 10_000), vocab_size=5000 + UNICODE_CHARS_SIZE
)
tokenizer_10000 = tokenizer.train_new_from_iterator(
    batch_iterator(dataset, 10_000), vocab_size=10_000 + UNICODE_CHARS_SIZE
)

tokenizer_1000.save_pretrained("tokenizer_1000/")
tokenizer_2000.save_pretrained("tokenizer_2000/")
tokenizer_5000.save_pretrained("tokenizer_5000/")
tokenizer_10000.save_pretrained("tokenizer_10000/")
