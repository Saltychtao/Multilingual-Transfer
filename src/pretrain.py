#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.


from accelerate.logging import get_logger
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    CONFIG_MAPPING
)

from src.trainer.trainer import Trainer
from src.models.modeling_gpt_neo import GPTNeoForCausalLM

from dataclasses import dataclass, field
from typing import Optional
import json

from src.data import make_data_module
from dataclasses import dataclass, field
from accelerate.utils import DistributedType


logger = get_logger(__name__)


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

@dataclass
class ModelArguments:
    model_type: Optional[str] = field(default="gpt_neox")
    model_name_or_path: Optional[str] = field(default=None)
    tokenizer_path: Optional[str] = field(default="")
    config_file: Optional[str] = field(default="")
    hf_config: Optional[str] = field(default=None)


@dataclass
class DataArguments:
    train_file: str = field(default=None, metadata={"help": "Path to the training data."})
    validation_file: str = field(default=None, metadata={"help": "Path to the training data."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    additional_save_steps: Optional[str] = field(default="1,2,4,8,16,32,64")
    def update(self, new):
        for key, value in new.items():
            if hasattr(self, key):
                setattr(self, key, value)


def get_model(model_args,model_config):
    if model_args.model_name_or_path is not None:
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path,trust_remote_code=True)
        print("Loading Pretrained Model from {}".format(model_args.model_name_or_path))
    else:
        config = AutoConfig.from_pretrained(model_args.hf_config,trust_remote_code=True)
        config.update(model_config)
        if config.architectures[0] == "GPTNeoForCausalLM":
            model = GPTNeoForCausalLM(config)
        else:
            model = AutoModelForCausalLM.from_config(config,trust_remote_code=True)
    return model

def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    with open(model_args.config_file) as f:
        config = json.load(f)
    training_args._frozen = False
    training_args.update(config["training_args"])
    if training_args.additional_save_steps != "":
        training_args.additional_save_steps = list(map(int,training_args.additional_save_steps.split(",")))
    else:
        training_args.additional_save_steps = []

    # print(training_args)
    # training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_path,trust_remote_code=True,padding_side="right")
    
    # special_tokens_dict = dict()
    # if tokenizer.pad_token is None:
    #     special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    # if tokenizer.eos_token is None:
    #     special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    # if tokenizer.bos_token is None:
    #     special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    # if tokenizer.unk_token is None:
    #     special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    config["model_args"]["vocab_size"] = len(tokenizer)
    model = get_model(model_args,config["model_args"])

    # smart_tokenizer_and_embedding_resize(
    #     special_tokens_dict=special_tokens_dict,
    #     tokenizer=tokenizer,
    #     model=model,
    # )
    # if tokenizer.pad_token_id is None or tokenizer.pad_token_id == 64000: # 64000 for baichuan model (older version)
    #     tokenizer.pad_token_id = 0 # set as the <unk> token
    data_module = make_data_module(data_args,model.config,tokenizer,type="lm")

    # Preprocessing the datasets.
    trainer = Trainer(model=model,tokenizer=tokenizer,args=training_args, **data_module)
    trainer.train()
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    main()