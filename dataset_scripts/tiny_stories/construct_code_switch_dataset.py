import json
import random

def main(args):
    with open(args.infile) as f:
        datas = [json.loads(line) for line in f]


    pairs = []
    with open(args.dict_file) as f:
        for line in f:
            srcword, tgtword = line.rstrip().split("\t")
            if srcword == tgtword:
                continue
            pairs.append((srcword,tgtword))

    

    
    