import argparse
import json

import random
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def build_data_module(data_file,tokenizer,src2tgt,tgt2src,ratio):

    def tokenize_function(examples):
        src_text, tgt_text = tokenizer.tokenize(examples["src_text"]), tokenizer.tokenize(examples["trg_text"])
        cs_src_text = []
        swap = 0
        prev_word = src_text[0]
        for i,s in enumerate(src_text[1:]):
            if s.startswith("▁"):
                prev_word =  prev_word.replace("▁","")
                if prev_word in src2tgt and random.random() < ratio:
                    cs_src_text.append(src2tgt[prev_word])
                    swap += 1
                else:
                    cs_src_text.append(prev_word)
                prev_word = s
            else:
                prev_word += s

        cs_tgt_text = []
        prev_word = tgt_text[0]
        for i,t in enumerate(tgt_text[1:]):
            if t.startswith("▁"):
                prev_word =  prev_word.replace("▁","")
                if prev_word in tgt2src and random.random() < ratio:
                    cs_tgt_text.append(tgt2src[prev_word])
                    swap += 1
                else:
                    cs_tgt_text.append(prev_word)
                prev_word = t
            else:
                prev_word += t

        return {
            "src_text": " ".join(src_text),
            "tgt_text": " ".join(tgt_text),
            "cs_src_text": " ".join(cs_src_text),
            "cs_tgt_text": " ".join(cs_tgt_text),
            "ratio": swap/len(src_text)
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

    return dataset["train"]


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path,trust_remote_code=True)
    src2tgt, tgt2src = {}, {}
    with open(args.dict_file) as f:
        for line in f:
            srcword, tgtword = line.split()
            if srcword == tgtword:
                continue
            src2tgt[srcword] = tgtword
            tgt2src[tgtword] = srcword

    dataset = build_data_module(args.infile,tokenizer,src2tgt,tgt2src,args.cs_ratio)

    ratio = sum(dataset["ratio"]) / len(dataset)
    print(ratio)
    final_datasets = dataset["src_text"] + dataset["tgt_text"] + dataset["cs_src_text"] + dataset["cs_tgt_text"]


    random.shuffle(final_datasets)

    with open(args.outfile + "{}.txt".format(round(ratio,2)),"w") as fout:
        for d in tqdm(final_datasets):
            fout.write(d + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile")
    parser.add_argument("--dict-file")
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--cs-ratio",default=0.0,type=float)
    parser.add_argument("--outfile")

    args = parser.parse_args()
    main(args)



                
    



