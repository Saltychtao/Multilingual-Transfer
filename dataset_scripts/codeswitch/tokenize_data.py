import json
from tqdm import tqdm
from sacremoses import MosesTokenizer, MosesPunctNormalizer
import string
import multiprocessing
from collections import defaultdict


with open("/mnt/bn/st-data-lq/jiahuanli/data/tiny_stories/en_fr/en_fr.json") as f:
    data = [json.loads(line) for line in f]

en_tokenizer, fr_tokenizer = MosesTokenizer("en"), MosesTokenizer("fr")
def tokenize(d):
    tokenized_en, tokenized_fr = en_tokenizer.tokenize(d["src_text"].lower()), fr_tokenizer.tokenize(d["trg_text"].lower())
    ret_en,ret_fr = [], []
    for w in tokenized_en:
        if w.endswith("."):
            ret_en.append(w[:-1])
            ret_en.append(".")
        else:
            ret_en.append(w)
    for w in tokenized_fr:
        if w.endswith("."):
            ret_fr.append(w[:-1])
            ret_fr.append(".")
        else:
            ret_fr.append(w)
    
    return " ".join(ret_en), " ".join(ret_fr)

pool = multiprocessing.Pool(processes=128)
tokenized = pool.map(tokenize,data)

with open("/mnt/bn/st-data-lq/jiahuanli/data/tiny_stories/en_fr/en_fr_tokenized.json","w") as f:
    for en, fr in tqdm(tokenized):
        f.write(
            json.dumps(
                {
                    "src_text": en,
                    "tgt_text": fr
                }, ensure_ascii=False
            ) + "\n"
        )