from transformers import AutoTokenizer
from datasets import load_dataset
from functools import partial
import pdb;
from tqdm import tqdm

def tokenize_function(tokenizer,examples):
    output = tokenizer(examples["text"])
    return output

tokenizer = AutoTokenizer.from_pretrained("/mnt/bn/st-data-lq/jiahuanli/models/Baichuan2-7B-Chat/",trust_remote_code=True)
tokenize_fn = partial(tokenize_function,tokenizer)
raw_dataset = load_dataset("json",data_files={"train":"/opt/tiger/fake_arnold/minipile/pure.json"})

tokenized_datasets = raw_dataset.map(
    tokenize_fn,
    batched=True,
    num_proc=120,
    load_from_cache_file=True,
    desc="Running tokenizer on dataset",
)

total = 0
for d in tqdm(tokenized_datasets["train"]):
    total += len(d["input_ids"])
print(total)