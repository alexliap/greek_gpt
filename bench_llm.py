import pickle
import warnings

import polars as pl

from benchmarks.val_data_llm import bench_llm
from model.language_model import load_model

warnings.filterwarnings("ignore")

tokenizer = "tokenizer_5000"

with open(f"data/tokenized_data/test_{tokenizer}_data.pkl", "rb") as file:
    test_data = pickle.load(file)

non_moe_greek_gpt = load_model("trained_models/model_8000_3.578_3.459/", is_moe=True)

benchmark_data = {"llm": [], "ce": [], "ppl": []}

ce, ppl = bench_llm(non_moe_greek_gpt, test_data)

benchmark_data["llm"].append("non_moe_gpt_11e6")
benchmark_data["ce"].append(ce)
benchmark_data["ppl"].append(ppl)

pl.from_dict(benchmark_data).write_csv("non_moe_llm_benchamrks.csv")
