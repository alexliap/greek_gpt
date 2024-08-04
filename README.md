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

## Results

Results can be found at the `benchmarks/results` directory. The MoE Transformer didn't seem to be superior from the Dense one, which was expected, but it wasn't faster either as everyone has mentioned.


<div align="center">

|         |  CE    | PPL    | Inference Time (for 800 tokens) |
| :-:     | :---:  | :---:  | :---:                           |
| MoE     | 3.629  | 37.659 | ~47 seconds                     |
| Dense   | 3.616  | 37.184 | ~32 seconds                     |

</div>

The speed advantage of MoE architectures most propably comes from the networking tricks applied when training a model on GPU clusters, rather than a single GPU.
