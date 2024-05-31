import pickle
import warnings

import polars as pl

from benchmarks.n_grams import load_ngram
from benchmarks.val_data_ngram import bench_ngram

warnings.filterwarnings("ignore")

tokenizer = "tokenizer_5000"

with open(f"data/tokenized_data/test_{tokenizer}_data.pkl", "rb") as file:
    test_data = pickle.load(file)

bi_gram = load_ngram("benchmarks/challenger_models/2-gram_8000_5.358_5.245/")
tri_gram = load_ngram("benchmarks/challenger_models/3-gram_8000_4.966_4.921/")
four_gram = load_ngram("benchmarks/challenger_models/4-gram_8000_4.82_4.735/")
five_gram = load_ngram("benchmarks/challenger_models/5-gram_8000_4.643_4.675/")

benchmark_data = {"ngram": [], "ce": [], "ppl": []}
ngram_list = [bi_gram, tri_gram, four_gram, five_gram]
for ngram in ngram_list:
    ce, ppl = bench_ngram(ngram, test_data)

    benchmark_data["ngram"].append(ngram.context_len + 1)
    benchmark_data["ce"].append(ce)
    benchmark_data["ppl"].append(ppl)

pl.from_dict(benchmark_data).write_csv("ngram_benchamrks.csv")
