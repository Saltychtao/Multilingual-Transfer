from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import json
import torch
from tqdm import tqdm

def load_dict(dict_file):
    dict_src_words, dict_tgt_words = set(), set()
    with open(dict_file) as f:
        for line in f:
            src,tgt = line.strip().split("\t")
            dict_src_words.add(src)
            dict_tgt_words.add(tgt)
    return dict_src_words, dict_tgt_words

def load_dataset(fname):
    sents = []
    with open(fname) as f:
        for line in f:
            sents.append(json.loads(line)["text"])
    return sents
        
def extract_error_word(tokenizer,ground_truth,predicted):
    error_words = []
    idx = 0
    cur = [ground_truth[0]]
    error_in_words = predicted[0] != ground_truth[0]
    import pdb; pdb.set_trace()
    while idx < len(ground_truth):
        if tokenizer.convert_ids_to_tokens(ground_truth[idx]).startswith("_"):
            if error_in_words:
                error_words.append(cur)
            cur = [ground_truth[idx]]
        else:
            cur.append(ground_truth[idx])
        error_in_words = predicted[idx] != ground_truth[idx]
        idx += 1
 
    return error_words


def main(args):
    tokenizer, model = AutoTokenizer.from_pretrained(args.model_path,trust_remote_code=True), AutoModelForCausalLM.from_pretrained(args.model_path,trust_remote_code=True)
    model = model.eval().cuda()

    sents = load_dataset(args.test_file)
    correct, total = 0,0
    error_words = []
    for sent in tqdm(sents):
        input_ids = tokenizer(sent,return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = model(**input_ids)
            logits = output.logits

            predicted = logits[:,:-1].argmax(dim=-1)
            ground_truth = input_ids["input_ids"][:,1:]
            correct += predicted.eq(ground_truth).sum().item()
            total += predicted.size(0) * predicted.size(1)
            error_words.extend(extract_error_word(tokenizer,ground_truth[0],predicted[0]))

    print("Word Prediction Accuracy: {}".format(correct/total))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path")
    parser.add_argument("--test-file")

    args = parser.parse_args()
    main(args)


