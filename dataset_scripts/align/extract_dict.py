import json
import sys
from tqdm import tqdm
from transformers import AutoTokenizer

with open(sys.argv[1]) as f:
    data = [json.loads(line) for line in tqdm(f)]


threshold = 0.9
src2tgt = {}
tgt2src = {}
tokenizer = AutoTokenizer.from_pretrained("/opt/tiger/fake_arnold/tokenizer",trust_remote_code=True)
for d in tqdm(data):
    src,tgt = d["src_text"], d["tgt_text"]
    src_words, tgt_words = src.split(), tgt.split()
    for a_s,a_t in zip(d["align_src_idx"],d["align_tgt_idx"]):
        src_word = src_words[a_s]
        tgt_word = tgt_words[a_t]
        if src_word not in src2tgt:
            src2tgt[src_word] = {}
        if tgt_word not in tgt2src:
            tgt2src[tgt_word] = {}

        if tgt_word in src2tgt[src_word]:
            src2tgt[src_word][tgt_word] += 1
        else:
            src2tgt[src_word][tgt_word] = 1

        if src_word in tgt2src[tgt_word]:
            tgt2src[tgt_word][src_word] += 1
        else:
            tgt2src[tgt_word][src_word] = 1

src2tgt_dict = set()
for src in src2tgt:
    tgt2num = src2tgt[src]
    total = sum(tgt2num.values())
    max_num = max(tgt2num.values())
    if max_num / total > threshold:
        tgt = max(tgt2num, key=tgt2num.get)
        src2tgt_dict.add((src,tgt))

tgt2src_dict = set()
for tgt in tgt2src:
    src2num = tgt2src[tgt]
    total = sum(src2num.values())
    max_num = max(src2num.values())
    if max_num / total > threshold:
        src = max(src2num,key=src2num.get)
        tgt2src_dict.add((src,tgt))

final_dict = src2tgt_dict & tgt2src_dict

with open(sys.argv[2] + ".tsv","w") as fout:
    for s,t in final_dict:
        fout.write(s + "\t" + t + "\n")