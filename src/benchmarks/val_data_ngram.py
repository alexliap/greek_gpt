import warnings

import mlx.core as mx
import mlx.nn as nn
from tqdm import tqdm

from benchmarks.n_grams import NGram

warnings.filterwarnings("ignore")


def bench_ngram(model: NGram, val_data):
    model.train(False)
    cross_entropy = 0
    log_prob_sum = 0
    total_preds = 0
    bar = tqdm(val_data, bar_format="{l_bar}{bar:15}{r_bar}{bar:-15b}")
    for sentence in bar:
        x = mx.array(
            [
                sentence[i : i + model.context_len]
                for i in range(0, len(sentence) - model.context_len)
            ]
        )
        y = mx.array(
            [
                sentence[i + model.context_len]
                for i in range(0, len(sentence) - model.context_len)
            ]
        )

        out = model(x)
        total_preds += nn.losses.cross_entropy(out, y).shape[0]
        cross_entropy += mx.sum(nn.losses.cross_entropy(out, y))

        probs = nn.softmax(out)
        log_prob_sum += mx.sum(mx.log(mx.take(probs, y, axis=1).diag()))

        model_config = f"{model.context_len + 1}-Gram: "
        ce_string = (
            f"Mean Cross Entropy: {round((cross_entropy / total_preds).item(), 3)}"
        )
        ppl_string = f"Mean PPL: {round(mx.exp(-log_prob_sum / total_preds).item(), 3)}"

        bar.set_description(model_config + ce_string + " | " + ppl_string)

    return round((cross_entropy / total_preds).item(), 3), round(
        mx.exp(-log_prob_sum / total_preds).item(), 3
    )
