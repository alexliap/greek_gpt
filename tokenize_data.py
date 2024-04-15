from transformers import AutoTokenizer
import pickle
import os
from tqdm import tqdm
import mlx.core as mx

os.environ["TOKENIZERS_PARALLELISM"] = "true"

with open("data/text_data/train_text_dataset.pkl", 'rb') as file:
    train_data = pickle.load(file)
with open("data/text_data/test_text_dataset.pkl", 'rb') as file:
    test_data = pickle.load(file)

if not os.path.exists("data/"):
    os.mkdir("data/")
if not os.path.exists("data/tokenized_data/"):
    os.mkdir("data/tokenized_data/")

tokenizer = AutoTokenizer.from_pretrained("custom_tokenizer/")

train_tokenized_data = []
for text in tqdm(train_data, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
    train_tokenized_data.append(tokenizer(text)['input_ids'])
with open("data/tokenized_data/train_tokenized_data.pkl", 'wb') as file:
    pickle.dump(train_tokenized_data, file)

test_tokenized_data = []
for text in tqdm(test_data, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
    test_tokenized_data.append(tokenizer(text)['input_ids'])
with open("data/tokenized_data/test_tokenized_data.pkl", 'wb') as file:
    pickle.dump(test_tokenized_data, file)
