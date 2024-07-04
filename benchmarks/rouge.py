import pickle
import warnings

from rouge import Rouge
from tqdm import tqdm
from transformers import AutoTokenizer

from model.language_model import load_model

warnings.filterwarnings("ignore")


def calc_rouge_score(tokenizer_dir: str, data_path: str, model_dir: str):
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

    rouge_scorer = Rouge()

    rouge_1 = []
    rouge_2 = []
    rouge_l = []
    for idx in tqdm(idxs, bar_format="{l_bar}{bar:15}{r_bar}{bar:-15b}"):
        query = tokenizer(data[idx])["input_ids"][:256]
        query = tokenizer.decode(query)

        sample_rouge_1 = 0
        sample_rouge_2 = 0
        sample_rouge_l = 0
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

            scores = rouge_scorer.get_scores(hyps=hypothesis, refs=reference)[0]

            sample_rouge_1 += scores["rouge-1"]["f"]
            sample_rouge_2 += scores["rouge-2"]["f"]
            sample_rouge_l += scores["rouge-l"]["f"]

        rouge_1.append(sample_rouge_1 / 10)
        rouge_2.append(sample_rouge_2 / 10)
        rouge_l.append(sample_rouge_l / 10)

    return {"rouge-1": rouge_1, "rouge-2": rouge_2, "rouge-l": rouge_l}
