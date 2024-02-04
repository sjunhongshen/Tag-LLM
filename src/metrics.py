"""
Metrics for gist token.
"""
import functools
from typing import Callable, Dict, Optional, Tuple

import nltk
import numpy as np
import pandas as pd
import torch
from transformers import PreTrainedTokenizer
from transformers.trainer_utils import EvalPrediction

from .arguments import Arguments

Metrics = Dict[str, float]
PredAndLogprobsOutput = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

# Must load after torch for some reason
import evaluate  # noqa

from datasets import DatasetDict


def strip_special_tokens(s):
    return (
        s.replace("<pad> ", "")
        .replace("</s>", "")
        .replace("<pad>", "")
        .replace("⁇", "")
        .strip()
    )


def nested_select(datasets: DatasetDict, max_len: int, **kwargs):
    return DatasetDict(
        {
            k: v.select(range(min(max_len, len(v))), **kwargs)
            for k, v in datasets.items()
        }
    )


def postprocess_text(preds, labels, remove_llama_padding=False):
    if remove_llama_padding:
        # XXX: This is a temporary hack because skip_special_tokens doesn't
        # seem to be working with the Llama SentencePiece tokenizer?
        preds = [pred.replace("⁇", "") for pred in preds]
        labels = [pred.replace("⁇", "") for pred in labels]

    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def compute_metrics(
    eval_preds: EvalPrediction,
    gist_token: int,
    tokenizer: PreTrainedTokenizer,
    args: Arguments,
    output_file: Optional[str] = None,
) -> Metrics:
    del gist_token

    preds = eval_preds.predictions
    labels = eval_preds.label_ids
    if isinstance(preds, tuple):
        preds = preds[0]

    results = {}

    # Compute ROUGE-L by decoding
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    is_llama = any(
        t in args.model.model_name_or_path.lower() for t in ("llama", "alpaca")
    )
    decoded_preds, decoded_labels = postprocess_text(
        decoded_preds,
        decoded_labels,
        remove_llama_padding=is_llama,
    )

    rouge_results = evaluate.load("rouge").compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    rouge_results = {k: round(v * 100, 4) for k, v in rouge_results.items()}
    results.update(rouge_results)

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    results["gen_len"] = np.mean(prediction_lens)

    if output_file is not None:
        if not hasattr(eval_preds, "inputs"):
            raise RuntimeError("If writing to output file, need inputs")
        inputs = np.where(eval_preds.inputs == -100, 0, eval_preds.inputs)
        decoded_inputs = tokenizer.batch_decode(inputs)
        decoded_inputs = list(map(strip_special_tokens, decoded_inputs))
        decoded_preds = list(map(strip_special_tokens, decoded_preds))
        decoded_labels = list(map(strip_special_tokens, decoded_labels))
        pd.DataFrame(
            {
                "x": decoded_inputs,
                "y_pred": decoded_preds,
                "y_true": decoded_labels,
            }
        ).to_csv(output_file, index=False)

    return results


def get_compute_metrics_fn(
    special_tokens: int, tokenizer: PreTrainedTokenizer, args: Arguments
) -> Callable:
    return functools.partial(
        compute_metrics, special_tokens=special_tokens, tokenizer=tokenizer, args=args
    )

