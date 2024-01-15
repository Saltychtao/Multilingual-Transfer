from argparse import Namespace
from functools import partial
from sre_constants import GROUPREF
from datasets import load_dataset
from accelerate import Accelerator
import torch
from itertools import chain
from typing import Any,  Dict, NewType,Sequence
import transformers
import datasets
from src.data.translation_prompter import Prompter
from src.trainer.synthetic.synthetic_converter import build_converter

from dataclasses import dataclass, field

def is_empty(fname):
    with open(fname) as f:
        content = f.read()
        return len(content) <= 1

@dataclass
class DataCollatorForLMDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        return_dict = {}
        for key in ("input_ids","labels"):
            if key not in instances[0]:
                continue
            entry = [torch.tensor(instance[key]).long() for instance in instances]
            data = torch.nn.utils.rnn.pad_sequence(
                entry, batch_first=True, padding_value=self.tokenizer.pad_token_id,)
            if "labels" in key:
                data[data.eq(self.tokenizer.pad_token_id)] = -100
            return_dict[key] = data
        return_dict["attention_mask"] = return_dict["input_ids"].ne(self.tokenizer.pad_token_id)
        return return_dict

@dataclass
class DataCollatorForParallelDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        entry = []
        for key in ("english_input_ids","chinese_input_ids"):
            if key not in instances[0]:
                continue
            entry.extend([torch.tensor(instance[key]).long() for instance in instances])
        data = torch.nn.utils.rnn.pad_sequence(
            entry, batch_first=True, padding_value=self.tokenizer.pad_token_id,)
        return_dict = {
            "input_ids": data,
            "attention_mask": data.ne(self.tokenizer.pad_token_id),
            "labels": data.masked_fill(data.eq(self.tokenizer.pad_token_id),-100)
        }
        return return_dict

def tokenize_parallel_concat_function(tokenizer,examples):
    english_prompts, chinese_prompts = Prompter.prompt_sentences(examples["src_text"],examples["tgt_text"])
    tokenized_english = tokenizer(english_prompts)
    tokenized_chinese = tokenizer(chinese_prompts)

    return {
        "input_ids": tokenized_english["input_ids"] + tokenized_chinese["input_ids"],
    }

def tokenize_parallel_function(tokenizer,examples):
    tokenized_english = tokenizer(examples["src_text"])
    tokenized_chinese = tokenizer(examples["tgt_text"])

    return {
        "english_input_ids": tokenized_english["input_ids"],
        "chinese_input_ids": tokenized_chinese["input_ids"],
    }

def tokenize_function(tokenizer,example):
    return tokenizer(example["text"])

def tokenize_synthesize_function(converter,tokenizer,example):
    return converter.convert_list(tokenizer(example["text"]))

def freq_score_function(codeswitch_token,example):
    input_id = example["input_ids"]
    need_codeswitch = 0
    for i in input_id:
        if i in codeswitch_token:
            need_codeswitch += 1
    freq_score = need_codeswitch / len(input_id)
    return {
        "input_ids": example["input_ids"],
        "attention_mask": example["attention_mask"],
        "freq_score": freq_score
    }

def group_texts(block_size,examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples["input_ids"])
    # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


            
def build_dataset(files,tokenize_fn,group_fn=None,freq_score_fn=None):
    ret_datasets = []
    remove_columns = []
    for file in files:
        if is_empty(file):
            continue
        dataset_args = {}
        raw_datasets = load_dataset("json", data_files={"train":file},**dataset_args)
        column_names = raw_datasets["train"].column_names

        dataset = raw_datasets.map(
            tokenize_fn,
            batched=True,
            num_proc=64,
            remove_columns=column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
        )
        if freq_score_fn is not None:
            dataset = dataset.map(
                freq_score_fn,
                batched=False,
                num_proc=64,
                load_from_cache_file=True,
                desc="Computing Freq Scores"
            )
            dataset = dataset.sort("freq_score",reverse=True)
            dataset = dataset.remove_columns("freq_score")
        if group_fn is not None:
            dataset = dataset.map(
                group_fn,
                batched=True,
                num_proc=64,
                load_from_cache_file=True,
                desc="Grouping texts in chunks of 1024",
            )

        ret_datasets.append(dataset["train"])

    if len(ret_datasets) == 0:
        print("No available content from {}".format(files))
        return None
    else:
        return datasets.concatenate_datasets(ret_datasets)


def make_lm_with_synthesis_module(data_args,model_args,tokenizer):
    converter = build_converter(data_args.converter_name,len(tokenizer))
    tokenize_fn = partial(tokenize_function,tokenizer)
    tokenize_synthesize_fn = partial(tokenize_synthesize_function,converter,tokenizer)

    group_fn = partial(group_texts,model_args.max_position_embeddings)
    if data_args.sort_by_freq_score == "1":
        freq_score_fn = partial(freq_score_function,data_args.codeswitch_token)
    else:
        freq_score_fn = None

    lm_datasets = build_dataset(data_args.train_file.split(","),tokenize_fn,group_fn,freq_score_fn)

    valid_english_file = data_args.validation_file

    valid_english_dataset = build_dataset([valid_english_file],tokenize_fn,group_fn)
    valid_synthesize_dataset = build_dataset([valid_english_file],tokenize_synthesize_fn,group_fn)
    
    train_dataset = lm_datasets
    lm_data_collator = DataCollatorForLMDataset(tokenizer=tokenizer)

    return dict(train_dataset=train_dataset, eval_dataset={"en":valid_english_dataset, "syn": valid_synthesize_dataset}, data_collator=lm_data_collator)



# model_args = Namespace(max_position_embeddings=512)
# data_args = Namespace(train_file="paracrawl.clean.json",validation_file=None)
# tokenizer = transformers.AutoTokenizer.from_pretrained("/mnt/bn/st-data-lq/jiahuanli/models/Baichuan2-13B-base",trust_remote_code=True)

# make_lm_data_module(data_args,model_args,tokenizer)

