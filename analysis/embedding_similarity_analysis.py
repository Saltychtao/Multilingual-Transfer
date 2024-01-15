from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse

def load_dict_with_only_one_token(dict_file,tokenizer):
    src_idx, tgt_idx = [],[]
    with open(dict_file) as f:
        for line in f:
            src,tgt = line.strip().split("\t")
            src_id, tgt_id = tokenizer(src)["input_ids"], tokenizer(tgt)["input_ids"]
            if len(src_id) == 1 and len(tgt_id) == 1:
                src_idx.append(src_id)
                tgt_idx.append(tgt_id)
    return src_idx, tgt_idx


def main(args):
    tokenizer, model = AutoTokenizer.from_pretrained(args.model_path,trust_remote_code=True), AutoModelForCausalLM.from_pretrained(args.model_path,trust_remote_code=True)
    model = model.eval()

    embed_tokens = model.get_input_embeddings().weight.data
    output_tokens = model.get_output_embeddings().weight.data

    src_idx, tgt_idx = load_dict_with_only_one_token(args.dict_file,tokenizer)
    src_idx, tgt_idx = torch.tensor(src_idx).squeeze(), torch.tensor(tgt_idx).squeeze()
    src_idx, tgt_idx = src_idx, tgt_idx

    src_embed, tgt_embed = embed_tokens[src_idx], embed_tokens[tgt_idx]
    src_proj, tgt_proj = output_tokens[src_idx], output_tokens[tgt_idx]

    align_similarity = torch.einsum("ih,ih->i",(src_embed,tgt_embed)).mean()
    random_similarity = torch.einsum("ih,jh->ij",(src_embed,tgt_embed[torch.randperm(len(src_embed))])).mean()
    print(align_similarity, random_similarity)

    align_similarity = torch.einsum("ih,ih->i",(src_proj,tgt_proj)).mean()
    random_similarity = torch.einsum("ih,jh->ij",(src_proj,tgt_proj[torch.randperm(len(src_proj))])).mean()
    print(align_similarity, random_similarity)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path")
    parser.add_argument("--dict-file")

    args = parser.parse_args()
    main(args)


