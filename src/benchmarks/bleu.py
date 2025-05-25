import pickle
import warnings

from sacrebleu import BLEU
from tqdm import tqdm
from transformers import AutoTokenizer

from model.language_model import load_model

warnings.filterwarnings("ignore")


def calc_bleu_score(tokenizer_dir: str, data_path: str, model_dir: str):
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    # load data
    with open(data_path, "rb") as file:
        data = pickle.load(file)
    # load model
    moe = load_model(model_dir, is_moe=True)
    # find the first 50 examples with more than 400 tokens
    idxs = []
    for i in range(len(data)):
        tokenized_text = tokenizer(data[i])["input_ids"]
        if len(tokenized_text) > 400:
            idxs.append(i)

        if len(idxs) == 50:
            break

    bleu_scorer = BLEU()

    scores = []
    for idx in tqdm(idxs, bar_format="{l_bar}{bar:15}{r_bar}{bar:-15b}"):
        query = tokenizer(data[idx])["input_ids"][:256]
        query = tokenizer.decode(query)

        sample_score = 0
        for i in range(10):
            completion = moe.generate(
                query=query, tokenizer=tokenizer, max_tokens=144, temperature=0.55
            )
            generated_query = tokenizer.decode(
                tokenizer(completion)["input_ids"][-144:]
            )

            hypothesis = generated_query
            reference = tokenizer.decode(
                tokenizer(data[idx])["input_ids"][256 : 256 + 144]
            )

            sample_score += bleu_scorer.sentence_score(
                hypothesis=hypothesis, references=[reference]
            ).score

        scores.append(sample_score / 10)

    return scores
