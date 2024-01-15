from functools import partial
from datasets import load_dataset
import torch
from itertools import chain
from typing import Any,  Dict, NewType,Sequence
import transformers
import datasets
import random
from collections import defaultdict

from src.trainer.trainer_utils import collate_tokens

from dataclasses import dataclass, field

def is_empty(fname):
    with open(fname) as f:
        content = f.read()
        return len(content) <= 1

@dataclass
class DataCollatorForLMDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    codeswitch_ratio: float
    codeswitch_table: Dict[tuple,tuple]
    codeswitch_corpus_ratio: float
    
    def is_start(self,i):
        return self.tokenizer.convert_ids_to_tokens(i).startswith("▁") or i == 0

    def codeswitch(self,input_ids):        
        codeswitched_ids = []
        codeswitched_align_mask = []
        original_align_mask = []
        cur = [input_ids[0]]
        for i in input_ids[1:] + [0]:
            if self.is_start(i):
                # cur word finished
                original_word_seq = cur
                if random.random() < self.codeswitch_ratio and tuple(original_word_seq) in self.codeswitch_table:
                    codeswitch_word_seq = random.choice(self.codeswitch_table[tuple(original_word_seq)])
                    codeswitched_ids.extend(codeswitch_word_seq)
                    codeswitched_align_mask.extend([0]*(len(codeswitch_word_seq)-1) + [1])
                    original_align_mask.extend([0]*(len(original_word_seq)-1) + [1])
                else:
                    codeswitched_ids.extend(original_word_seq)
                    codeswitched_align_mask.extend([1]*(len(original_word_seq)))
                    original_align_mask.extend([1]*(len(original_word_seq)))
                cur = [i]
            else:
                cur.append(i)
        assert len(original_align_mask) == len(input_ids)
        assert len(codeswitched_align_mask) == len(codeswitched_ids)
        return {
            "codeswitched_ids": codeswitched_ids,
            "original_ids": input_ids,
            "codeswitched_align_mask": codeswitched_align_mask,
            "original_align_mask": original_align_mask
        }

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        return_dict = {}
        if self.codeswitch_corpus_ratio == 0:
            input_ids = [torch.tensor(instance["input_ids"]).long() for instance in instances]
            input_ids = collate_tokens(input_ids, self.tokenizer.pad_token_id, left_pad=False)
            return {
                "input_ids": input_ids,
                "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
                "labels": input_ids.masked_fill(input_ids.eq(self.tokenizer.pad_token_id), -100),
            }
        else:
            codeswitch_corpus_size = int(self.codeswitch_corpus_ratio*len(instances))
            lm_corpus = instances[codeswitch_corpus_size:]
            codeswitched_corpus = instances[:codeswitch_corpus_size]

            lm_input_ids = [torch.tensor(instance["input_ids"]).long() for instance in lm_corpus]
            
            codeswitched = [self.codeswitch(instance["input_ids"]) for instance in codeswitched_corpus]
            codeswitched_ids = [torch.tensor(codeswitch["codeswitched_ids"]).long() for codeswitch in codeswitched]
            codeswitched_align_mask = [torch.tensor(codeswitch["codeswitched_align_mask"]).long() for codeswitch in codeswitched]
            original_input_ids = [torch.tensor(codeswitch["original_ids"]).long() for codeswitch in codeswitched]
            original_align_mask = [torch.tensor(codeswitch["original_align_mask"]).long() for codeswitch in codeswitched]

            input_ids = original_input_ids + codeswitched_ids + lm_input_ids
            align_mask = original_align_mask + codeswitched_align_mask

            input_ids = collate_tokens(input_ids, self.tokenizer.pad_token_id, left_pad=False)
            align_mask = collate_tokens(align_mask,0, left_pad=False)

            return_dict = {
                "input_ids": input_ids,
                "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
                "labels": input_ids.masked_fill(input_ids.eq(self.tokenizer.pad_token_id), -100),
                "align_mask": align_mask
            }
        return return_dict

@dataclass
class DataCollatorForParallelDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    sentswitch_ratio: float

    def is_punc(self,i):
        return self.tokenizer.convert_ids_to_tokens(i) in ["▁.","▁!","▁?"]

    def segment_to_sentences(self,input_ids):
        sentences = []
        cur = [input_ids[0]]
        for i in input_ids[1:]:
            cur.append(i)
            if self.is_punc(i):
                sentences.append(cur)
                cur = []
        return sentences

    def sentswitch(self,src_input_ids,tgt_input_ids):
        src_sentences, tgt_sentences = self.segment_to_sentences(src_input_ids), self.segment_to_sentences(tgt_input_ids)
        if len(src_sentences) != len(tgt_sentences):
            sentswitched_ids =  src_input_ids if random.random() > 0.5 else tgt_input_ids
            original_ids = sentswitched_ids
            return {
                "sentswitched_ids": sentswitched_ids,
                "sentswitched_masks": [0] * len(sentswitched_ids),
                "original_masks": [0] * len(original_ids),
                "original_ids": original_ids
            }
        else:
            sentswitched_ids, sentswitch_masks, original_masks = [], [], []
            for src_sent, tgt_sent in zip(src_sentences,tgt_sentences):
                if random.random() < self.sentswitch_ratio:
                    sentswitched_ids.extend(tgt_sent)
                    sentswitch_masks.extend([0]*len(tgt_sent))
                    original_masks.extend([0]*len(src_sent))
                else:
                    sentswitched_ids.extend(src_sent)
                    sentswitch_masks.extend([1]*len(src_sent))
                    original_masks.extend([1]*len(src_sent))
        return {
            "sentswitched_ids": sentswitched_ids,
            "sentswitched_masks": sentswitch_masks,
            "original_masks": original_masks,
            "original_ids": src_input_ids
        }


    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        return_dict = {}
        src_input_ids, tgt_input_ids = [instance["x"] for instance in instances], [instance["y"] for instance in instances]
        sentswitched = [self.sentswitch(src_input_ids[i],tgt_input_ids[i]) for i in range(len(src_input_ids))]

        src_input_ids = [torch.tensor(x).long() for x  in src_input_ids]
        tgt_input_ids = [torch.tensor(x).long() for x in tgt_input_ids]

        sentswitched_ids = [torch.tensor(x["sentswitched_ids"]).long() for x in sentswitched]
        sentswitched_masks = [torch.tensor(x["sentswitched_masks"]).long() for x in sentswitched]
        original_ids = [torch.tensor(x["original_ids"]).long() for x in sentswitched]
        original_masks = [torch.tensor(x["original_masks"]).long() for x in sentswitched]

        input_ids = original_ids + sentswitched_ids + tgt_input_ids
        masks = original_masks + sentswitched_masks

        src_and_tgt_input_ids = collate_tokens(src_input_ids + tgt_input_ids,pad_idx=self.tokenizer.pad_token_id,left_pad=False)

        input_ids = collate_tokens(input_ids,pad_idx=self.tokenizer.pad_token_id,left_pad=False)
        masks = collate_tokens(masks,pad_idx=0,left_pad=False)

        return_dict = {
            "x_and_y": {
                "input_ids": src_and_tgt_input_ids,
                "attention_mask": src_and_tgt_input_ids.ne(self.tokenizer.pad_token_id),
                "labels": src_and_tgt_input_ids.masked_fill(src_and_tgt_input_ids.eq(self.tokenizer.pad_token_id),-100)
            },
            "sentswitch":{
                "input": {
                    "input_ids": input_ids,
                    "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
                    "labels": input_ids.masked_fill(input_ids.eq(self.tokenizer.pad_token_id),-100)
                },
                "masks": masks,
            }
        }

        return return_dict

def tokenize_parallel_function(tokenizer,examples):
    tokenized_english = tokenizer(examples["src_text"])
    tokenized_chinese = tokenizer(examples["tgt_text"])

    return {
        "x": [s+ [tokenizer.eos_token_id] for s in tokenized_english["input_ids"]],
        "y": [t+ [tokenizer.eos_token_id] for t in tokenized_chinese["input_ids"]],
        "xy": [s + [tokenizer.eos_token_id] + t + [tokenizer.eos_token_id] for s,t in zip(tokenized_english["input_ids"],tokenized_chinese["input_ids"])],
        "yx": [t + [tokenizer.eos_token_id] + s + [tokenizer.eos_token_id] for s,t in zip(tokenized_english["input_ids"],tokenized_chinese["input_ids"])],
        "xx": [s + [tokenizer.eos_token_id] + s + [tokenizer.eos_token_id] for s,t in zip(tokenized_english["input_ids"],tokenized_chinese["input_ids"])],
        "yy": [t + [tokenizer.eos_token_id] + t + [tokenizer.eos_token_id] for s,t in zip(tokenized_english["input_ids"],tokenized_chinese["input_ids"])]
    }

def tokenize_function(tokenizer,example):
    return {
        "input_ids": [s+ [tokenizer.eos_token_id] for s in tokenizer(example["text"])["input_ids"]]
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


            
def build_dataset(files,tokenize_fn,group_fn=None):
    ret_datasets = []
    for file in files:
        if file == "none" or is_empty(file) :
            continue
        dataset_args = {}
        raw_datasets = load_dataset("json", data_files={"train":file}, **dataset_args)
        column_names = raw_datasets["train"].column_names

        dataset = raw_datasets.map(
            tokenize_fn,
            batched=True,
            num_proc=64,
            remove_columns=column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
        )
        if group_fn is not None:
            dataset = dataset.map(
                group_fn,
                batched=True,
                num_proc=64,
                load_from_cache_file=True,
                desc="Grouping texts in chunks of 1024"
            )

        ret_datasets.append(dataset["train"])

    if len(ret_datasets) == 0:
        print("No available content from {}".format(files))
        return None
    else:
        return datasets.concatenate_datasets(ret_datasets)


def make_lm_with_parallel_module(data_args,model_args,tokenizer):
    tokenize_parallel_fn = partial(tokenize_parallel_function,tokenizer)
    tokenize_fn = partial(tokenize_function,tokenizer)
    group_fn = partial(group_texts,model_args.max_position_embeddings)

    lm_datasets = build_dataset(data_args.lm_train_file.split(","),tokenize_fn,group_fn)

    valid_english_file, valid_chinese_file = data_args.validation_file.split(",")
    parallel_datasets = build_dataset([data_args.parallel_train_file],tokenize_parallel_fn,None)

    valid_english_dataset = build_dataset([valid_english_file],tokenize_fn,group_fn)
    valid_chinese_dataset = build_dataset([valid_chinese_file], tokenize_fn,group_fn)
    
    if data_args.codeswitch_ratio > 0:
        codeswitch_table = load_codeswitch_table_from_file(data_args.dict_file,tokenizer)
    else:
        codeswitch_table = None

    lm_data_collator = DataCollatorForLMDataset(tokenizer=tokenizer,codeswitch_table=codeswitch_table,codeswitch_ratio=data_args.codeswitch_ratio,codeswitch_corpus_ratio=data_args.codeswitch_corpus_ratio)
    valid_lm_data_collator = DataCollatorForLMDataset(tokenizer=tokenizer,codeswitch_table=codeswitch_table,codeswitch_ratio=0.,codeswitch_corpus_ratio=0.)
    parallel_data_collator = DataCollatorForParallelDataset(tokenizer=tokenizer,sentswitch_ratio=data_args.sentswitch_ratio)

    return dict(train_dataset={"lm":lm_datasets,"parallel":parallel_datasets}, eval_dataset={"en":valid_english_dataset,"fr":valid_chinese_dataset}, data_collator=(lm_data_collator,parallel_data_collator,valid_lm_data_collator))


def load_codeswitch_table_from_file(dict_file,tokenizer):
    print("Loading dict from {}".format(dict_file))
    codeswitch_table = defaultdict(lambda: [])
    with open(dict_file) as f:
        for line in f:
            src_word, tgt_word = tuple(line.strip().split("\t"))
            src_word_seq = tuple(tokenizer(src_word)["input_ids"])
            tgt_word_seq = tokenizer(tgt_word)["input_ids"]
            codeswitch_table[src_word_seq].append(tgt_word_seq)
    return codeswitch_table

# tokenizer = transformers.AutoTokenizer.from_pretrained("/mnt/bn/st-data-lq/jiahuanli/data/tiny_stories/en_fr/en_fr_tokenizer_bpe4000",trust_remote_code=True)
# codeswitch_table = load_codeswitch_table_from_file("dict.txt",tokenizer)
# lm_data_collator = DataCollatorForLMDataset(tokenizer=tokenizer,codeswitch_ratio=1.0,codeswitch_table=codeswitch_table)

# input_ids = tokenizer("What a good day and goodcastle!")["input_ids"]
# lm_data_collator.codeswitch(input_ids)


# parallel_data_collator = DataCollatorForParallelDataset(tokenizer=tokenizer,sentswitch_ratio=1.0)
# src_doc = "What a good day! I like it very much!"
# tgt_doc = "What an aweful day. I hate it very much!"

# src_input_ids = tokenizer(src_doc)["input_ids"]
# tgt_input_ids = tokenizer(tgt_doc)["input_ids"]

# parallel_data_collator.sentswitch(src_input_ids,tgt_input_ids)
# model_args = Namespace(max_position_embeddings=512)
# data_args = Namespace(train_file="paracrawl.clean.json",validation_file=None)
# tokenizer = transformers.AutoTokenizer.from_pretrained("/mnt/bn/st-data-lq/jiahuanli/models/Baichuan2-13B-base",trust_remote_code=True)

# make_lm_data_module(data_args,model_args,tokenizer)

