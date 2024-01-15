import json
import random
import argparse
import os
from tqdm import tqdm
def save_jsonl(data,filename):
    with open(filename,"w") as fout:
        for d in data:
            fout.write(json.dumps(d,ensure_ascii=False) + "\n")

def main(args):
    datas = []
    with open(args.infile) as f:
        datas = [json.loads(line) for line in f]
    ratios = [float(t) for t in args.ratios.split(",")]
    valid, train = datas[:10000], datas[10000:]
    valid_en = [{"text":d["src_text"]} for d in valid]
    valid_zh = [{"text":d["tgt_text"]} for d in valid]

    os.makedirs(args.savedir,exist_ok=True)

    save_jsonl(valid_en,args.savedir+"/valid_{}.jsonl".format(args.srclang))
    save_jsonl(valid_zh,args.savedir+"/valid_{}.jsonl".format(args.tgtlang))

    length = len(train)
    en = [{"text": d["src_text"]} for d in train[:int(ratios[0]*length)]]
    if args.with_overlap:
        zh = [{"text":d["tgt_text"]} for d in train[:int(ratios[1]*length)]]
    else:
        zh = [{"text":d["tgt_text"]} for d in train[int(ratios[0]*length):int((ratios[0]+ratios[1])*length)]]
    parallel = [
        {
            "src_text": d["src_text"],
            "tgt_text": d["tgt_text"]
        } for d in train[-int(ratios[2]*length+1):]
    ]
    print("En: {}".format(len(en)))
    print("Fr: {}".format(len(zh)))
    print("Parallel: {}".format(len(parallel)))
    save_jsonl(en,args.savedir+"/{}.jsonl".format(args.srclang))
    save_jsonl(zh,args.savedir+"/{}.jsonl".format(args.tgtlang))
    save_jsonl(parallel,args.savedir+"/parallel.jsonl")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile")
    parser.add_argument("--savedir")
    parser.add_argument("--ratios")
    parser.add_argument("--srclang")
    parser.add_argument("--tgtlang")
    parser.add_argument("--split-for-valid",default=False,action="store_true")
    parser.add_argument("--with-overlap",default=False,action="store_true")

    args = parser.parse_args()
    main(args)