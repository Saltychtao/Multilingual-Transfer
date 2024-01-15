import argparse
import json
from tqdm import tqdm
import numpy as np

def main(args):
    cur_text = []
    json_data = []
    lengths = []
    skipped = 0
    total = 0
    with open(args.infile) as f:
        for line in tqdm(f):
            if line.strip() == "":
                continue
            elif line.strip() == "<|endoftext|>":
                total += 1
                doc = " ".join(cur_text)
                cur_text = []

                length = len(doc.split())
                if length > 256:
                    skipped += 1
                    continue
                json_data.append(
                    {"text":doc}
                )
                lengths.append(len(doc.split()))
            else:
                cur_text.append(line.strip())

    print(skipped,total)
    with open(args.savefile,"w") as fout:
        for d in json_data:
            fout.write(json.dumps(d,ensure_ascii=False) + "\n")

    print(np.histogram(lengths))
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile")
    parser.add_argument("--savefile")

    args = parser.parse_args()
    main(args)


