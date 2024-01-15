from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from evaluation.tasks import build_prompter
import torch
from tqdm import tqdm

import argparse

def read_data(infile):
    data = []
    with open(infile) as f:
        for line in f:
            data.append(json.loads(line))
    return data

def main(args):
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,trust_remote_code=True).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,trust_remote_code=True)

    pool_data = read_data(args.pool_file)
    test_data = read_data(args.test_file)

    prompter = build_prompter(args.task)

    correct = 0
    for d in tqdm(test_data):
        prompt = prompter.construct_few_shot(pool_data,d)
        encoding = tokenizer(prompt,return_tensors="pt").to("cuda")
        with torch.no_grad():
            out_ids = model.generate(
                **encoding,
                max_new_tokens=256,
                num_beams=4,
                early_stopping=True
            )
            import pdb;pdb.set_trace()

            completion = tokenizer.batch_decode(out_ids,skip_special_tokens=True)[0]
            label = prompter.extract_label(prompt,completion)
            if label == d["label"]:
                correct += 1

    print("Accuracy: {}".format(correct / len(test_data)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-file")
    parser.add_argument("--test-file")
    parser.add_argument("--model-name-or-path")
    parser.add_argument("--task")

    args = parser.parse_args()
    main(args)











