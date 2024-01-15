from datasets import load_dataset
import sys
from transformers import AutoTokenizer
from collections import Counter
from tqdm import tqdm
import torch

def build_data_module(data_file,tokenizer):

    def tokenize_function(examples):
        tokenized_inputs = tokenizer(examples["text"])
        input_ids = tokenized_inputs["input_ids"]
        return {
            "input_ids": input_ids
        }

    raw_datasets = load_dataset("json", data_files={"train":data_file},**{})
    column_names = raw_datasets["train"].column_names

    dataset = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=120,
        remove_columns=column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )
    return dataset["train"]

tokenizer = AutoTokenizer.from_pretrained("../tokenizer",trust_remote_code=True)
dataset = build_data_module(sys.argv[1],tokenizer)
counter = Counter()
for d in tqdm(dataset):
    counter.update(d["input_ids"])

total = sum(counter.values())
frequencies = []
for i in range(len(tokenizer)):
    if i not in counter:
        frequencies.append(0)
    else:
        frequencies.append(counter[i] / total)

torch.save(frequencies,sys.argv[2])
