# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Tag training script, adapted from huggingface's run_clm.py example.
"""

import logging
import os
import pickle
import hydra
import torch 
import torch.nn as nn
import numpy as np
from omegaconf.dictconfig import DictConfig
from transformers import (
    AutoConfiger,
    AutoTokenizer,
    LlamaTokenizer,
    is_torch_tpu_available,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from . import tag_llama
from .arguments import Arguments, global_setup
from .collator import DataCollatorForTagLLM
from .tag_llama import DEBUG_LLAMA_CONFIG, TagLlamaForCausalLM
from .integrations import CustomWandbCallback, EvaluateFirstStepCallback
from .metrics import get_compute_metrics_fn, nested_select
from .trainer_seq2seq import TagSeq2SeqTrainer
from .get_data import get_dataset

from peft import get_peft_model, LoraConfig, PromptTuningConfig, PromptEncoderConfig, TaskType, PromptTuningInit

# Will error if the minimal version of Transformers is not installed. Remove at
# your own risks.
check_min_version("4.28.0.dev0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt",
)

logger = logging.getLogger(__name__)
torch.cuda.empty_cache()

NORM_RATIO = 7.8

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )
    
@hydra.main(config_path="conf", config_name="config")
def main(args: DictConfig) -> None:
    args: Arguments = global_setup(args)

    # Detect last checkpoint
    last_checkpoint = None
    if (
        os.path.isdir(args.training.output_dir)
        and args.training.do_train
        and not args.training.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(args.training.output_dir)
        if last_checkpoint is None and len(os.listdir(args.training.output_dir)) > 0:
            existing_files = os.listdir(args.training.output_dir)
            logger.warning(
                (
                    "Output directory (%s) already exists and "
                    "is not empty. Existing files: %s. "
                    "Training anyways as these may just be output files."
                ),
                args.training.output_dir,
                str(existing_files),
            )
        elif (
            last_checkpoint is not None and args.training.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To "
                "avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from "
                "scratch."
            )

    # Set seed before initializing model
    set_seed(args.training.seed)

    config_kwargs = {
        "cache_dir": args.model.cache_dir,
        "revision": args.model.model_revision,
        "use_auth_token": True if args.model.use_auth_token else None,
        }

    if args.model.llama_debug:
        if args.model.pretrained:
            raise RuntimeError("llama_debug requires pretrained set to False")
        config = DEBUG_LLAMA_CONFIG
    elif args.model.config_name:
        config = AutoConfig.from_pretrained(args.model.config_name, **config_kwargs)
    elif args.model.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.model.model_name_or_path, **config_kwargs
        )
    else:
        raise ValueError(
            "Unlike run_clm.py, this script does not support specifying a model type "
            "from scratch. Specify args.model.model_name_or_path and set "
            "args.pretrained = False to train from scratch instead."
        )


    ###### Tag setup ######

    task_name = args.model.task_name    

    # Load pretrained special token dict and weights
    if args.model.tag_dict_path is not None:
        with open(args.model.tag_dict_path + '/tag_name_dict.pkl', 'rb') as f:
            tag_name_dict = pickle.load(f)

        embedding_weights = torch.from_numpy(np.load(args.model.tag_dict_path + "/embedding_weights.npy")).float()
        num_existing_tokens = embedding_weights.shape[0]
    else:
        tag_name_dict = {}
        embedding_weights = None
        num_existing_tokens = 0
        
    freeze = args.model.freeze_existing_tags
    
    if args.model.peft:
        args.model.regression = False
        args.model.use_domain_tag = False
        args.model.use_function_tag = False
        args.model.add_ce_loss = False
        args.model.autoregressive_attn_mask = True

    train_dataset, eval_dataset, tag_name_dict, num_new_tokens, tags_to_update, domain_tags = get_dataset(task_name, num_existing_tokens, tag_name_dict, args.model.num_token_per_tag, args.model.use_domain_tag, args.model.use_function_tag, args.model.regression, freeze, True)
    config.update({"num_new_tokens": num_new_tokens, "output_dir": args.training.output_dir, "regression_out_dim": args.model.regression_out_dim})
    print("Output dir:", args.training.output_dir, "\nTag dict:", tag_name_dict, "\nTags to learn:", tags_to_update, "\nTrain data size:", len(train_dataset))
    
    if not os.path.exists(args.training.output_dir):
        os.makedirs(args.training.output_dir)
    with open(args.training.output_dir + "/tag_name_dict.pkl", 'wb') as f:
        pickle.dump(tag_name_dict, f)
    with open(args.training.output_dir + "/arguments.pkl", 'wb') as f:
        pickle.dump(args, f)


    ###### Tokenizer ######

    tokenizer_kwargs = {
        "cache_dir": args.model.cache_dir,
        "use_fast": args.model.use_fast_tokenizer,
        "revision": args.model.model_revision,
        "use_auth_token": True if args.model.use_auth_token else None,
    }

    if args.model.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model.tokenizer_name, **tokenizer_kwargs
        )
    elif args.model.model_name_or_path:
        tokenizer = LlamaTokenizer.from_pretrained(
                args.model.model_name_or_path, **tokenizer_kwargs
            )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported "
            "by this script."
            "You can do it from another script, save it, and load it from here, using "
            "--tokenizer_name."
        )

    model_cls = TagLlamaForCausalLM
    
    if args.model.pretrained:
        model = model_cls.from_pretrained(
            args.model.model_name_or_path,
            config=config,
            cache_dir=args.model.cache_dir,
            revision=args.model.model_revision,
            use_auth_token=True if args.model.use_auth_token else None,
            load_in_8bit=True,
            device_map="auto"
            )
    else:
        model = model_cls(config)

    avg_emb = model.model.embed_tokens.weight.data.mean(0).clone()
    
    if args.model.peft:
        model.word_embeddings = model.model.embed_tokens
        if args.model.peft == "lora":
            peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.05)
        elif args.model.peft == "prompttuning":
            if task_name == "DC" or task_name == "QED" or task_name == "Descriptor": 
                num_prompt = 2
            elif task_name == "BA":
                num_prompt = 3
            else:
                num_prompt = 1
                
            peft_config = PromptTuningConfig(
                    task_type=TaskType.CAUSAL_LM,
                    prompt_tuning_init=PromptTuningInit.RANDOM,
                    num_virtual_tokens=args.model.num_token_per_tag * num_prompt,
                    tokenizer_name_or_path=args.model.model_name_or_path,
                )
        if args.model.peft != "linearprobe":
            model = get_peft_model(model, peft_config)
           
        if args.model.regression:
            if args.model.peft == "lora":
                model.base_model.model.lm_head_reg = nn.Linear(config.hidden_size, args.model.regression_out_dim, bias=False)
                model.base_model.model.lm_head_reg.weight = nn.Parameter(torch.stack([avg_emb] * args.model.regression_out_dim))
            elif args.model.peft == "prompttuning":
                model.base_model.lm_head_reg = nn.Linear(config.hidden_size, args.model.regression_out_dim, bias=False)
                model.base_model.lm_head_reg.weight = nn.Parameter(torch.stack([avg_emb] * args.model.regression_out_dim))
    else:
        ###### Embedding weight initialization ######
    
        model.model.augmented_embedder.embedding.weight.data = nn.Parameter(torch.stack([avg_emb * NORM_RATIO] * num_new_tokens))
    
        if embedding_weights is not None:
            if not freeze:
                model.model.augmented_embedder.embedding.weight.data = model.model.augmented_embedder.embedding.weight.data.to(avg_emb.dtype)
                embedding_weights=embedding_weights.to(model.model.augmented_embedder.embedding.weight.device)
                model.model.augmented_embedder.embedding = nn.Embedding.from_pretrained(torch.cat([embedding_weights, model.model.augmented_embedder.embedding.weight.data])).to(avg_emb.dtype)
            else:
                model.model.augmented_embedder.original_embedder.weight.data = model.model.augmented_embedder.original_embedder.weight.data.to(avg_emb.dtype)
                embedding_weights=embedding_weights.to(model.model.augmented_embedder.original_embedder.weight.device)
                model.model.augmented_embedder.original_embedder = nn.Embedding.from_pretrained(torch.cat([model.model.augmented_embedder.original_embedder.weight.data, embedding_weights ])).to(avg_emb.dtype)
    
        if freeze:
            model.model.augmented_embedder.vocab_size = model.model.augmented_embedder.original_embedder.weight.data.shape[0]
            model.model.augmented_embedder.added_tokens = [model.model.augmented_embedder.vocab_size + i for i in range(num_new_tokens)]
        else:
            model.model.augmented_embedder.added_tokens = [model.model.augmented_embedder.vocab_size + i for i in range(num_existing_tokens + num_new_tokens)]
     

    ###### Regression input/output initialization ######

    if args.model.regression:
        model.lm_head_reg = nn.Linear(config.hidden_size, args.model.regression_out_dim, bias=False)
        model.lm_head_reg.weight = nn.Parameter(torch.stack([avg_emb] * args.model.regression_out_dim))     

    if (not args.model.peft):
        # Freeze original weight
        for name, module in model.named_children():
            for param in module.parameters():
                param.requires_grad = False

        # Set update parameters
        for param in model.model.augmented_embedder.embedding.parameters():
            param.requires_grad = True
    
        if args.model.regression:
            for param in model.lm_head_reg.parameters():
                param.requires_grad = True
                    
    elif args.model.peft == "linearprobe":
        for name, module in model.named_children():
            for param in module.parameters():
                param.requires_grad = False
        for param in model.lm_head_reg.parameters():
            param.requires_grad = True
    
    # Check grad setting
    for name, module in model.named_children():
        for n, param in module.named_parameters():
            if param.requires_grad:
                print(name, n, param.shape, param)

    # Check if special token has already been added to the model (e.g. because
    # we're resuming from a checkpoint.)
    
    if len(tokenizer) != tag_llama.PRETRAINED_VOCAB_SIZE + num_new_tokens:
        # Add tag to tokenizer
        tokenizer.add_special_tokens({"additional_special_tokens": ["<TAG " + str(i) + ">" for i in range(num_existing_tokens + num_new_tokens)]})        
        
    special_tokens = tokenizer.additional_special_tokens_ids

    if args.training.do_train:
        if args.data.max_train_samples is not None:
            print("Truncate training dataset to size", args.data.max_train_samples)
            max_train_samples = min(len(train_dataset), args.data.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if args.training.do_eval:
        
        if args.data.max_eval_samples is not None:
            eval_dataset = nested_select(
                eval_dataset,
                args.data.max_eval_samples,
            )

        compute_metrics = get_compute_metrics_fn(
            special_tokens=special_tokens, tokenizer=tokenizer, args=args
        )

    print_trainable_parameters(model)
    
    ###### Data setting ######
    domain_tags = np.array(domain_tags) + tag_llama.PRETRAINED_VOCAB_SIZE
    
    data_collator = DataCollatorForTagLLM(
            tokenizer,
            max_length=1024,
            pad_token=tokenizer.pad_token_id,
            pretrained_vocab_size=tag_llama.PRETRAINED_VOCAB_SIZE,
            check_correctness=True,
            tag_name_dict=tag_name_dict,
            tags_to_update=tags_to_update,
            domain_tags=domain_tags,
            num_token_per_tag=args.model.num_token_per_tag,
            use_function_tag = args.model.use_function_tag,
            add_ce_loss=args.model.add_ce_loss,
            autoregressive_attn_mask=args.model.autoregressive_attn_mask
                )
    
    ###### Trainer ######

    custom_callbacks = []
    if args.wandb.log:
        custom_callbacks.append(CustomWandbCallback(args))
    if args.training.evaluate_before_train:
        custom_callbacks.append(EvaluateFirstStepCallback())

    trainer = TagSeq2SeqTrainer(
        model=model,
        args=args.training,
        train_dataset=train_dataset if args.training.do_train else None,
        eval_dataset=eval_dataset if args.training.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
        if args.training.do_eval and not is_torch_tpu_available()
        else None,
        preprocess_logits_for_metrics=None,
        callbacks=custom_callbacks,
    )

    if args.training.fp16:
        trainer.scaler = torch.cuda.amp.GradScaler(init_scale=2.**14)

    ###### Training ######

    if args.training.do_train:
        checkpoint = None
        if args.training.resume_from_checkpoint is not None:
            checkpoint = args.training.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            args.data.max_train_samples
            if args.data.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        if not args.model.peft:
            torch.save(model.model.augmented_embedder.state_dict(), args.training.output_dir + "/augmented_embedder.pth")
            np.save(args.training.output_dir + "/embedding_weights.npy", model.model.augmented_embedder.embedding.weight.data.clone().detach().cpu().numpy())
        else:
            model.save_pretrained(args.training.output_dir)
            
        try:
            torch.save(model.lm_head_reg.state_dict(), args.training.output_dir + "/lm_head_reg.pth")
            np.save(args.training.output_dir + "/lm_head_reg.npy", model.lm_head_reg.weight.data.detach().cpu().numpy())
        except:
            torch.save(model.base_model.model.lm_head_reg.state_dict(), args.training.output_dir + "/lm_head_reg.pth")
            np.save(args.training.output_dir + "/lm_head_reg.npy", model.base_model.model.lm_head_reg.weight.data.detach().cpu().numpy())


if __name__ == "__main__":
    main()