#!/bin/bash

# in order to be able to run the script first run "chmod +x pipeline.sh"

source .venv/bin/activate

python ./cli/preprocessing/make_data.py

echo "Saved Greek version of Wikipedia locally in a text file."

python ./cli/preprocessing/train_tokenizer.py --vocab-size 5000

echo "Trained 4 new GPT2 tokenizers from scratch with sizes: 1000 | 2000 | 5000 | 10000."

python ./cli/preprocessing/tokenize_data.py --vocab-size 5000

echo "Tokenized the dataset with all tokenizers, resulting in 4 versions."
