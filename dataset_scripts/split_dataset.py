import json
import random
import argparse

def save_jsonl(data,filename):
    with open(filename,"w") as fout:
        for d in data:
            fout.write(json.dumps(d,ensure_ascii=True) + "\n")

def main(args):
    datas = []
    with open(args.infile) as f:
        datas = [json.loads(line) for line in f]

    A_start, A_end = 0, (len(datas) - args.num_bilingual) // 2
    B_start, B_end = A_end, len(datas) - args.num_bilingual
    C_start = B_end

    random.shuffle(datas)
    A_set = datas[A_start:A_end]
    B_set = datas[B_start:B_end]
    C_set = datas[C_start:]

    save_jsonl(A_set,args.savedir+"/train.A.jsonl")
    save_jsonl(B_set,args.savedir+"/train.B.jsonl")
    save_jsonl(C_set,args.savedir+"/train.C.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile")
    parser.add_argument("--savedir")
    parser.add_argument("--num-bilingual",type=int,default=200000)

    args = parser.parse_args()
    main(args)





