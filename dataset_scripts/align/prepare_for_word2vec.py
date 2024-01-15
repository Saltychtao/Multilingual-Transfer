import json
import sys

with open(sys.argv[1]) as f:
    data = [json.loads(line) for line in f]

with open(sys.argv[2] + ".en","w") as fout:
    for d in data:
        fout.write(d["src_text"] + "\n")

with open(sys.argv[2] + ".fr","w") as fout:
    for d in data:
        fout.write(d["tgt_text"] + "\n")