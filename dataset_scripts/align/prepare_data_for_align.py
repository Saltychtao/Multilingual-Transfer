import json
import sys
from mosestokenizer import MosesTokenizer
from tqdm import tqdm

with open(sys.argv[1]) as f:
    data = [json.loads(line) for line in f]

with MosesTokenizer('en') as tokenize_en, MosesTokenizer("fr") as tokenize_fr:
    with open(sys.argv[2],"w") as fout:
        for d in tqdm(data):
            fout.write(" ".join(tokenize_en(d["src_text"])) + " ||| " + " ".join(tokenize_fr(d["trg_text"])) + "\n")
