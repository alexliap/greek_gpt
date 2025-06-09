#!/bin/bash

# in order to be able to run the script first run "chmod +x pipeline.sh"

source .venv/bin/activate

python ./cli/preprocessing/make_data.py

echo "Saved Greek version of Wikipedia locally in a text file."

python ./cli/preprocessing/train_tokenizer.py --vocab-size 5000

echo "Trained new GPT2 tokenizer from scratch."

python ./cli/preprocessing/tokenize_data.py --vocab-size 5000

echo "Tokenized the dataset with the selected tokenizer."
