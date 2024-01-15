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


from lib2to3.pgen2.token import OP
from optparse import Option
from accelerate.logging import get_logger
import transformers
from transformers import (
    AutoModelForCausalLM,
    GPTNeoXForCausalLM,
    AutoTokenizer,
    AutoConfig,
    CONFIG_MAPPING
)
import torch

from src.trainer.synthetic.synthetic_trainer import SyntheticTrainer
from src.models.modeling_gpt_neo import GPTNeoForCausalLM
from src.models.baichuan.modeling_baichuan import BaichuanForCausalLM
from src.models.modeling_llama import LlamaForCausalLM

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


@dataclass
class ModelArguments:
    model_type: Optional[str] = field(default="gpt_neox")
    model_name_or_path: Optional[str] = field(default=None)
    tokenizer_path: Optional[str] = field(default="")
    config_file: Optional[str] = field(default="")
    hf_config: Optional[str] = field(default=None)

    bind_embedding: Optional[str] = field(default=None)
    bind_embedding_mode: Optional[str] = field(default="all")

    pretrained_word_embedding_en: Optional[str] = field(default=None)
    pretrained_word_embedding_syn: Optional[str] = field(default=None)


@dataclass
class DataArguments:
    train_file: str = field(default=None, metadata={"help": "Path to the training data."})
    validation_file: str = field(default=None, metadata={"help": "Path to the training data."})

    sort_by_freq_score: str = field(default=None)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    dataloader_shuffle: Optional[bool] = field(default=True)
    additional_save_steps: Optional[str] = field(default="1,2,4,8,16,32,64")
    cons_strength: Optional[float] = field(default=0)
    cons_layers: Optional[str] = field(default=None)
    cons_patterns: Optional[str] = field(default=None)
    cons_metrics: Optional[str] = field(default=None)

    contrastive_tau: Optional[float] = field(default=None)
    contrastive_p: Optional[float] = field(default=2)

    converter_name: Optional[str] = field(default="one2one")
    synthesis_ratio: Optional[str] = field(default="0.8,0.18,0.02")

    codeswitch_ratio: Optional[float] = field(default=0.)
    codeswitch_dict_tokens: Optional[float] = field(default=0)
    codeswitch_label_prediction_mode: Optional[str] = field(default="codeswitch")

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
        if config.architectures[0] == "LlamaForCausalLM":
            model = LlamaForCausalLM(config)
        else:
            model = AutoModelForCausalLM.from_config(config,trust_remote_code=True)
    return model

def bind_embedding(model,vocab_size,bind_embedding_mode,topk):
    print("Binding Embedding")
    if bind_embedding_mode == "all":
        start,end=  0, vocab_size
    elif bind_embedding_mode == "topk":
        start, end = 0, topk
    elif bind_embedding_mode == "lastk":
        start, end = topk+1, vocab_size
    embed_tokens = model.get_input_embeddings()
    for i in range(start,end):
        embed_tokens.weight.data[i+vocab_size] = embed_tokens.weight.data[i].clone()

    output_tokens = model.get_output_embeddings()
    for i in range(vocab_size):
        output_tokens.weight.data[i+vocab_size] = output_tokens.weight.data[i].clone()

def load_from_pretrained_embedding(model,tokenizer, pretrained_embedding_path,offset=False):
    with open(pretrained_embedding_path) as f:
        lines = f.readlines()
        vocab_size, dim = tuple(lines[0].split())
        vocab_size, dim = int(vocab_size), int(dim)
        token2dict = {}
        for line in lines[1:]:
            splited = line.split()
            token = splited[0]
            if len(splited[1:]) != dim:
                continue
            embeddings = list(map(float,splited[1:]))
            token2dict[token] = torch.tensor(embeddings)
        mean_embeddings = sum(token2dict.values()) / len(token2dict)


    embed_tokens = model.get_input_embeddings()
    output_tokens = model.get_output_embeddings()
    for i in range(vocab_size):
        token = tokenizer.convert_ids_to_tokens(i)
        embedding = token2dict.get(token,mean_embeddings)
        embed_tokens.weight.data[i+offset] = embedding
        output_tokens.weight.data[i+offset] = embedding

def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    with open(model_args.config_file) as f:
        config = json.load(f)
    training_args._frozen = False
    training_args.logging_nan_inf_filter = False
    training_args.update(config["training_args"])
    if training_args.additional_save_steps != "":
        training_args.additional_save_steps = list(map(int,training_args.additional_save_steps.split(",")))
    else:
        training_args.additional_save_steps = []

    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_path,trust_remote_code=True,padding_side="right")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0
    config["model_args"]["vocab_size"] = 2*len(tokenizer)
    model = get_model(model_args,config["model_args"])

    if model_args.bind_embedding == "1":
        bind_embedding(model,len(tokenizer),model_args.bind_embedding_mode, training_args.codeswitch_from_top_k)

    if model_args.pretrained_word_embedding_en:
        load_from_pretrained_embedding(model,tokenizer,model_args.pretrained_word_embedding_en,offset=0)
    if model_args.pretrained_word_embedding_syn:
        load_from_pretrained_embedding(model,tokenizer,model_args.pretrained_word_embedding_en,offset=len(tokenizer))

    if training_args.codeswitch_dict_tokens == 0:
        codeswitch_token = torch.tensor([]).long()
    elif training_args.codeswitch_dict_tokens < 0:
        codeswitch_token = torch.tensor(
            list(range(len(tokenizer)))[int(training_args.codeswitch_dict_tokens):]
    ).long()
    elif training_args.codeswitch_dict_tokens < 1:
        codeswitch_token = torch.rand((len(tokenizer))).lt(training_args.codeswitch_dict_tokens).nonzero().squeeze()
    else:
        
        codeswitch_token = torch.tensor(
            list(range(len(tokenizer)))[:int(training_args.codeswitch_dict_tokens)]
        ).long()

    training_args.codeswitch_token = codeswitch_token
    if data_args.sort_by_freq_score:
        data_args.codeswitch_token = set(codeswitch_token.tolist())
        training_args.sorted_dataset = True
    else:
        training_args.sorted_dataset = False
    data_args.converter_name = training_args.converter_name
    data_module = make_data_module(data_args,model.config,tokenizer,type="lm_with_synthesis")

    # Preprocessing the datasets.
    trainer = SyntheticTrainer(model=model,tokenizer=tokenizer,args=training_args, **data_module)
    trainer.train()
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    main()