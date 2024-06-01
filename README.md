# Greek GPT

This is a first attempt in making a MoE Language Model with only Greek text.
Implementation is done using the Apple MLX framework.

## Data

The data used for training and validation is [Greek Version of Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia/viewer/20231101.el), which consists of 227k URL links of Wikipedia text content.

## Models

Two transformer models were developed:

- One with MoE architecture with 10.5M parameters.
- One Dense transformer with 10.5M parameters.

Also some sohisticated NGrams were created and trained for comparison purposes.

- 2-Gram up to 5-Gram.

## Tokenizer

As far as the tokenizer is concerned, the GPT-2 one was used from Hugging Face and was retrained on our dataset. The vocab size was configured at 5000 tokens. No particular focus was given on the tokenizer training, but is an essential part of language modeling and is as important as the model itself.
