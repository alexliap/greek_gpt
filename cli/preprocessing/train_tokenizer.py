import argparse
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download folder from Azure")
    parser.add_argument("--vocab-size", required=True)
    args = parser.parse_args()

    vocab_size = int(args.vocab_size)

    tokenizer = tokenizer.train_new_from_iterator(
        batch_iterator(dataset, vocab_size), vocab_size=vocab_size + UNICODE_CHARS_SIZE
    )

    tokenizer.save_pretrained(f"tokenizer_{vocab_size}/")
