import json
from src.data.translation_prompter import Prompter
import argparse

def load_dataset(filepath):
    datas = []
    with open(filepath) as f:
        for line in f:
            datas.append(json.loads(line))
    return datas

def main(args):
    all_datas = load_dataset(args.parallel_datas)
    length = int(len(all_datas) * (1-args.ratio))

    unshuffled_parallel = []
    unshuffled_lm = []
    for i in range(0,length,1024):
        start,end = i, min(i+1024,length)
        src_sents = [d["src_text"] for d in all_datas[start:end]]
        tgt_sents = [d["tgt_text"] for d in all_datas[start:end]]
        english_translation_examples, chinese_translation_examples = Prompter.prompt_sentences(src_sents,tgt_sents)
        for d in all_datas[start:end]:
            unshuffled_parallel.append(d)
        if args.group:
            unshuffled_lm.append({"text":"\n".join(english_translation_examples)})
            unshuffled_lm.append({"text":"\n".join(chinese_translation_examples)})
        else:
            unshuffled_lm.extend([{"text":zh_d} for zh_d in chinese_translation_examples])
            unshuffled_lm.extend([{"text":en_d} for en_d in english_translation_examples])

    shuffled = []
    for d in all_datas[length:]:
        shuffled.append(
            {"text":d["src_text"].strip()}
            )
        shuffled.append(
            {"text":d["tgt_text"].strip()}
            )

    with open(args.savedir+"/parallel.shuffled_{}.json".format(args.ratio),"w") as fout:
        for d in shuffled:
            fout.write(json.dumps(d,ensure_ascii=False) + "\n")

    with open(args.savedir+"/parallel.unshuffled_{}.parallel.json".format(args.ratio),"w") as fout:
        for d in unshuffled_parallel:
            fout.write(json.dumps(d,ensure_ascii=False) + "\n")

    with open(args.savedir+"/parallel.unshuffled_{}.lm.json".format(args.ratio),"w") as fout:
        for d in unshuffled_lm:
            fout.write(json.dumps(d,ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel-datas")
    parser.add_argument("--ratio",type=float)
    parser.add_argument("--group",default=False,action="store_true")
    parser.add_argument("--savedir")

    args = parser.parse_args()
    main(args)        
