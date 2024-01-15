from ast import arg
import sentencepiece as spm
import argparse
import json
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile")
    parser.add_argument("--save-model-path")
    parser.add_argument("--vocab-size")

    args = parser.parse_args()

    with open(args.infile) as f, open("./tmp","w") as fout:
        datas = []
        for line in tqdm(f):
            d = json.loads(line)
            datas.append(d["src_text"])
            datas.append(d["trg_text"])

        for d in tqdm(datas):
            fout.write(d + "\n")

    spm.SentencePieceTrainer.train(input="./tmp", model_prefix=args.save_model_path, vocab_size=args.vocab_size, model_type="bpe",split_by_white_space=True)