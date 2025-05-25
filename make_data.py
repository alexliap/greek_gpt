import os

from datasets import load_dataset

from src.data_processing.txt_processing import create_txt_dataset

if not os.path.exists("data/"):
    os.mkdir("data/")
if not os.path.exists("data/text_data/"):
    os.mkdir("data/text_data/")

data = load_dataset("wikimedia/wikipedia", "20231101.el")

txt_dataset = create_txt_dataset(dataset=data)

with open("data/text_data/txt_dataset.txt", "w", encoding="utf-8") as file:
    file.write(txt_dataset)
