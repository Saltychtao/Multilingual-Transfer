import argparse
from collections import defaultdict
from transformers import AutoTokenizer
import torch
from tqdm import tqdm

def main(args):
    datas = []
    with open(args.infile) as f:
        for line in tqdm(f):
            src,tgt = tuple(line.strip().split(" ||| "))
            datas.append((src,tgt))

    aligns = []
    with open(args.alignfile) as f:
        for line in tqdm(f):
            cur_align = []
            for a in line.split():
                a_s,a_t = tuple(a.split("-"))
                a_s,a_t = int(a_s), int(a_t)
                cur_align.append((a_s,a_t))
            aligns.append(cur_align)

    src2tgt = {}
    tgt2src = {}
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path,trust_remote_code=True)
    for (src,tgt), align in tqdm(zip(datas,aligns)):
        src_words, tgt_words = src.split(), tgt.split()
        for a_s,a_t in align:
            # print(a_s,len(src_words))
            # print(a_t,len(tgt_words))
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

    import pdb; pdb.set_trace()
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile")
    parser.add_argument("--alignfile")
    parser.add_argument("--tokenizer-path")

    args = parser.parse_args()
    main(args)