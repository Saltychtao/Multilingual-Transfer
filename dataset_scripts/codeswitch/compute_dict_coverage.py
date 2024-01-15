import json
import transformers
from tqdm import tqdm
from sacremoses import MosesTokenizer, MosesPunctNormalizer
import string
import multiprocessing
from collections import defaultdict

align_dict = defaultdict(lambda: [])
dict_words_src, dict_words_tgt = set(), set()
with open("dict-all.txt") as f:
    for line in f:
        src,tgt = tuple(line.strip().split("\t"))
        dict_words_src.add(src)
        dict_words_tgt.add(tgt)



with open("/mnt/bn/st-data-lq/jiahuanli/data/tiny_stories/en_fr/en_fr_tokenized.json") as f:
    data = [json.loads(line) for line in f]

en_total, fr_total = 0, 0
en_found, fr_found = 0, 0
en_wordset, fr_wordset = set(), set()

for d in tqdm(data):
    for w in d["src_text"].split():
        en_total += 1
        en_wordset.add(w)
        if w in dict_words_src:
            en_found += 1

    for w in d["tgt_text"].split():
        fr_total += 1
        fr_wordset.add(w)
        if w in dict_words_tgt:
            fr_found += 1

print(en_found/en_total, fr_found/fr_total)

en_residual = en_wordset - dict_words_src
print(len(en_residual) / len(en_wordset))

fr_residual = fr_wordset - dict_words_tgt
print(len(fr_residual) / len(fr_wordset))








