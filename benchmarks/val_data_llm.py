import warnings

import mlx.core as mx
import mlx.nn as nn
from tqdm import tqdm

from model.language_model import LanguageModel

warnings.filterwarnings("ignore")


def bench_llm(model: LanguageModel, val_data):
    cross_entropy = 0
    total_preds = 0
    log_prob_sum = 0
    bar = tqdm(val_data, bar_format="{l_bar}{bar:15}{r_bar}{bar:-15b}")
    for sentence in bar:
        if len(sentence) < model.context_len:
            x = mx.array(sentence[:-1]).reshape(1, -1)
            y = mx.array(sentence[1:])

            out = model(x)
            total_preds += nn.losses.cross_entropy(out, y).shape[0]
            cross_entropy += mx.sum(nn.losses.cross_entropy(out, y))

            probs = nn.softmax(out)
            log_prob_sum += mx.sum(mx.log(mx.take(probs, y, axis=1).diag()))
        else:
            start_idxs = list(range(0, len(sentence) - 1, model.context_len - 1))
            end_idxs = list(
                range(model.context_len - 2, len(sentence) - 1, model.context_len - 1)
            )
            # corner case 415, where length is 1276
            if (len(sentence) - 1) % 255 != 0:
                end_idxs += [len(sentence) - 1]
            for start, end in zip(start_idxs, end_idxs):
                x = mx.array(sentence[start : end + 1][:-1]).reshape(1, -1)
                y = mx.array(sentence[start : end + 1][1:])

                out = model(x)
                total_preds += nn.losses.cross_entropy(out, y).shape[0]
                cross_entropy += mx.sum(nn.losses.cross_entropy(out, y))

                probs = nn.softmax(out)
                log_prob_sum += mx.sum(mx.log(mx.take(probs, y, axis=1).diag()))

        ce_string = (
            f"Mean Cross Entropy: {round((cross_entropy/total_preds).item(), 3)}"
        )
        ppl_string = f"Mean PPL: {round(mx.exp(-log_prob_sum/total_preds).item(), 3)}"

        bar.set_description(ce_string + " | " + ppl_string)

    return round((cross_entropy / total_preds).item(), 3), round(
        mx.exp(-log_prob_sum / total_preds).item(), 3
    )
