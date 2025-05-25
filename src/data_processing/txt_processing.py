from datasets.dataset_dict import DatasetDict
from tqdm import tqdm
from transformers import AutoTokenizer


def create_txt_dataset(dataset: DatasetDict) -> str:
    if "train" in dataset.keys():
        txt_dataset = ""
        dt = dataset["train"]
        for i in tqdm(range(len(dt)), bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}"):
            txt_dataset += dt[i]["text"]
            txt_dataset += " "
    else:
        raise RuntimeError(
            "There is no 'train' key in this dataset. Maybe you have to change the key."
        )

    return txt_dataset


def tokenize_txt_dataset(tokenizer: AutoTokenizer, dataset: DatasetDict) -> list[int]:
    if "train" in dataset.keys():
        tokenized_data = []
        dt = dataset["train"]
        for i in tqdm(range(len(dt)), bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}"):
            text = dt[i]["text"] + " "
            tokenized_text = tokenizer(text)["input_ids"]
            tokenized_data.append(tokenized_text)
    else:
        raise RuntimeError(
            "There is no 'train' key in this dataset. Maybe you have to change the key."
        )

    return tokenized_data
