import logging
from collections import defaultdict
from dataclasses import dataclass
from pickle import INST
from re import S
from typing import Optional, List

import itertools
import random
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerBase

from difflib import SequenceMatcher
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = logging.getLogger(__name__)


@dataclass
class DataCollatorForTagLLM:
    """Data collator for decoder-only models. Does left padding."""
    
    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    pad_token: int = 0
    pretrained_vocab_size: int = 32000
    tag_name_dict: dict = None
    domain_tags: List[str] = None
    tags_to_update: List[str] = None
    check_correctness: bool = False
    eval_mode: bool = False
    num_token_per_tag: int = 10
    use_function_tag: bool = True
    add_ce_loss: bool = False
    autoregressive_attn_mask: bool = False

    def __call__(self, batch, return_tensors=None):

        max_length = self.max_length

        if return_tensors is None:
            return_tensors = self.return_tensors

        model_inputs = defaultdict(list)

        reg_idx = []
        reg_pred_idx = None
        reg_dim = []
        clf_idx = []    
        tag_idx = []
        prompt_len = []
        ori_len = []
        
        num_passed = 0

        tokenized_function_tag = self.tokenizer(self.tag_name_dict[self.tags_to_update[-1]])["input_ids"][1:]

        for idx, instance in enumerate(batch):
            prompt_len.append([])

            instance_formulation = instance["formulation"]
            for tag_token in self.tags_to_update:
                instance_formulation = instance_formulation.replace(tag_token, self.tag_name_dict[tag_token])

            if isinstance(instance['input'], list):
                for ip_idx, ip in enumerate(instance['input']):
                    instance_formulation = instance_formulation.replace("<input " + str(ip_idx) +">", ip)
                    if ip_idx < 2:    
                        prompt_len[-1].append(len(self.tokenizer(ip)["input_ids"]) - 1)
            else:
                instance_formulation = instance_formulation.replace("<input>", instance["input"])
                prompt_len[-1].append(len(self.tokenizer(instance["input"])["input_ids"]) - 1)
            instance_formulation = instance_formulation.replace("<output>", str(instance["output"]))

            tokenized_input = self.tokenizer(instance_formulation)["input_ids"] + [self.tokenizer.eos_token_id]
            if instance["task"] == "Generation":
                tokenized_input = tokenized_input[1:-1]
            elif self.eval_mode:
                tokenized_input = tokenized_input[:-1]

            if instance["regression"]:
                label_reg = float(instance["output"])
                labels = [self.label_pad_token_id] * len(tokenized_input)

            if not instance["regression"] or self.add_ce_loss:
                labels = tokenized_input.copy()
                special_token_idx = np.argwhere((np.array(tokenized_input) >= self.pretrained_vocab_size) | (np.array(tokenized_input) == 0)).flatten()
                labels = np.array(labels)
                labels[special_token_idx] = self.label_pad_token_id
                labels = labels.tolist()

            if len(tokenized_input) > max_length:
                if num_passed == 0 and idx == len(batch) - 1:
                    to_trim = len(tokenized_input) - max_length
                    labels = labels[to_trim:]
                    tokenized_input = tokenized_input[to_trim:]
                else:
                    if instance["regression"] or self.eval_mode:
                        prompt_len = prompt_len[:-1]
                        continue

                    to_trim = len(tokenized_input) - max_length
                    labels = labels[to_trim:]
                    tokenized_input = tokenized_input[to_trim:]
            
            model_inputs["input_ids"].append(tokenized_input)
            model_inputs["labels"].append(labels)
            model_inputs["attention_mask"].append([1 for _ in tokenized_input])
            
            if instance["regression"]:
                if not self.use_function_tag:
                    reg_pred_idx = -2
                else:
                    try:
                        reg_pred_idx = np.argwhere(np.array(tokenized_input) == tokenized_function_tag[-1]).flatten()[-1] - len(tokenized_input) #
                    except:
                        reg_pred_idx = -2
                reg_idx.append(num_passed)
                reg_dim.append(instance["regression_dim"])
                model_inputs["label_reg"].append(label_reg)

            if not instance["regression"] or self.add_ce_loss:
                clf_idx.append(num_passed)
                            
            ori_len.append(len(tokenized_input))
            tag_idx.append(np.argwhere(np.in1d(np.array(tokenized_input), self.domain_tags)).flatten())
            num_passed += 1

        # Left-pad inputs, convert to tensor
        for key, value in model_inputs.items():
            if key == "label_reg":
                model_inputs[key] = torch.tensor(value).float()
            else:
                if key == "labels":
                    pad_token_id = self.label_pad_token_id
                elif key == "attention_mask":
                    pad_token_id = 0
                else:
                    pad_token_id = self.tokenizer.pad_token_id

                # To left-pad inputs, reverse, then right-pad, then reverse
                value_tensors = [torch.tensor(v[::-1]) for v in value]
                model_inputs[key] = torch.fliplr(
                    pad_sequence(
                        value_tensors,
                        batch_first=True,
                        padding_value=pad_token_id,
                    )
                )
        
        # mask for domain tags
        bs, max_len = model_inputs["input_ids"].shape
        attn_mask = torch.zeros(bs, max_len, max_len)
        
        if not self.autoregressive_attn_mask:
            for i, idx in enumerate(tag_idx):
                if len(idx) == 0: continue

                idx += max_len - ori_len[i]

                k = 0
                for j, x in enumerate(idx):
                    if x == 0 or len(prompt_len[i]) <= k:
                        continue
                    for y in range(x, x + self.num_token_per_tag + prompt_len[i][k]):
                        if y >= len(model_inputs["input_ids"][i]): break
                        attn_mask[i, y, :x] = 1
                    k += 1

            model_inputs["attention_mask_new"] = attn_mask.unsqueeze(1)

        if len(reg_idx) > 0:
            model_inputs["reg_idx"] = reg_idx
            model_inputs["reg_dim"] = reg_dim
            model_inputs["reg_pred_idx"] = reg_pred_idx
        if len(clf_idx) > 0:
            model_inputs["clf_idx"] = clf_idx
    
        return model_inputs

