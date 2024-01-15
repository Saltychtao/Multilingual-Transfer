from functools import partial
from transformers import AutoTokenizer
from datasets import load_dataset
import random
from itertools import chain
import torch
import transformers
from dataclasses import dataclass

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

def build_data_module(data_file,tokenizer,frequencies,args):

    def tokenize_function(examples):
        tokenized_inputs = tokenizer(examples["text"])
        input_ids = tokenized_inputs["input_ids"]
        new_input_ids = []
        for i in input_ids:
            if random.random() < frequencies[i]:
                continue
            else:
                new_input_ids.append(i)
        return {
            "input_ids": new_input_ids
        }

    raw_datasets = load_dataset("json", data_files={"train":data_file},**{})
    column_names = raw_datasets["train"].column_names

    dataset = raw_datasets.map(
        tokenize_function,
        batched=False,
        num_proc=128,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )
    group_fn = partial(group_texts,1024)
    dataset = dataset.map(
        group_fn,
        batched=True,
        num_proc=120,
        load_from_cache_file=True,
        desc="Grouping texts in chunks of 1024"
    )

    return dataset["train"], DataCollatorForParallelDataset(tokenizer)

@dataclass
class DataCollatorForParallelDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        entry = []
        for key in ("input_ids",):
            if key not in instances[0]:
                continue
            entry.extend([torch.tensor(instance[key]).long() for instance in instances])
        data = torch.nn.utils.rnn.pad_sequence(
            entry, batch_first=True, padding_value=self.tokenizer.pad_token_id,)
        return_dict = {
            "input_ids": data,
        }
        return return_dict