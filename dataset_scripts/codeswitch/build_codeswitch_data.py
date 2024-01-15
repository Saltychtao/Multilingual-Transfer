from collections import defaultdict
import json
import random
import multiprocessing
import argparse

def load_codeswitch_table_from_file(dict_file):
    print("Loading dict from {}".format(dict_file))
    codeswitch_table = defaultdict(lambda: [])
    with open(dict_file) as f:
        for line in f:
            src_word, tgt_word = tuple(line.strip().split("\t"))
            codeswitch_table[src_word].append(tgt_word)
    return codeswitch_table

align_dict = load_codeswitch_table_from_file("dict-all.txt")

def codeswitch(d):
    ret = []
    for w in d["text"].split():
        if w in align_dict:
            _w = random.choice(align_dict[w])
            ret.append(_w)
        else:
            ret.append(w)
    return json.dumps(
        {
            "text": " ".join(ret)
        }, ensure_ascii=False
    )

def main(args):
    
    with open(args.infile) as f:
        data = [json.loads(line) for line in f]

    pool = multiprocessing.Pool(processes=64)
    codeswitched = pool.map(codeswitch,data)

    with open(args.outfile,"w") as fout:
        fout.write("\n".join(codeswitched))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile")
    parser.add_argument("--dict-file")
    parser.add_argument("--outfile")

    args = parser.parse_args()
    main(args)
