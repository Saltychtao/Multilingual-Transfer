import argparse
import json
from tqdm import tqdm
import numpy as np

def main(args):
    data = []
    cur = []
    skipped = 0
    total = 0
    with open(args.infile) as f:
        for line in tqdm(f):
            if line.strip() == "":
                continue
            elif line.strip() == "<|endoftext|>":
                total += 1
                text = " ".join(cur)
                if len(text.split()) > 500:
                    skipped += 1
                    continue
                data.append(
                    {"text": text}
                )
                cur = []
            else:
                cur.append(line.strip())

    print(skipped, total)
    with open(args.savefile,"w") as fout:
        for d in data:
            fout.write(json.dumps(d,ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile")
    parser.add_argument("--savefile")

    args = parser.parse_args()
    main(args)


