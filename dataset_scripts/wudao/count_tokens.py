from this import d
from transformers import AutoTokenizer
import csv
import json
from tqdm import tqdm
import argparse

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,trust_remote_code=True)
    total_length = 0
    final_data = []
    for file in args.input_files:
        with open(file) as f:
            data = json.load(f)
        for d in tqdm(data):
            tokenized = tokenizer(d["content"])
            total_length += len(tokenized["input_ids"])
            final_data.append(
                {"text": d["content"]}
            )
            if total_length > 1600000:
                break
        if total_length > 1600000:
            break

    with open(args.savefile,"w") as fout:
        for d in tqdm(final_data):
            fout.write(json.dumps(d,ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser =argparse.ArgumentParser()
    parser.add_argument("--input-files",nargs="+")
    parser.add_argument("--model-path")
    parser.add_argument("--savefile")

    args = parser.parse_args()
    main(args)

        



