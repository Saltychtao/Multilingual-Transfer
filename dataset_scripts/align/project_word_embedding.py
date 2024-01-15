import json
import torch
import argparse
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer

def load_embeddings(filename):
    word2idx = {}
    embeddings = []
    with open(filename) as f:
        for i,line in tqdm(enumerate(f)):
            if i == 0:
                cnt, dim = tuple(line.split())
                cnt, dim = int(cnt), int(dim)
            else:
                word_and_emb = line.split()
                if len(word_and_emb) != dim + 1:
                    continue
                word = word_and_emb[0]
                emb = list(map(float,word_and_emb[1:]))
                embeddings.append(emb)
                word2idx[word] = i-1
    return word2idx, torch.tensor(embeddings).cuda()


def load_dict(filename):
    dict = []
    with open(filename) as f:
        for line in f:
            srcword,tgtword = line.split()
            srcword = "▁" + srcword
            tgtword = "▁" + tgtword
            dict.append((srcword,tgtword))
    return dict

def evaluate_NN(vectors,tokenizer,word_list):
    inputs = tokenizer(word_list,return_tensors="pt",padding=True)
    tokens = inputs["input_ids"].cuda().squeeze()
    query_embedding = torch.nn.functional.normalize(vectors[tokens],dim=-1)
    all_embedding = torch.nn.functional.normalize(vectors[torch.arange(0,len(tokenizer)-1).to(tokens)],dim=-1)
    dist = torch.einsum("ih,jh->ij",(query_embedding,all_embedding))

    for i, word in enumerate(word_list):
        nearest_ids = dist[i].topk(k=20,dim=-1).indices
        print(word, tokenizer.convert_ids_to_tokens(nearest_ids))


def main(args):
    src_word2idx, src_emb_matrix = load_embeddings(args.src_emb)
    tgt_word2idx, tgt_emb_matrix  = load_embeddings(args.tgt_emb)
    dict = load_dict(args.dict_file)

    src_align_emb = []
    tgt_align_emb = []
    for srcword, tgtword in dict:
        if srcword not in src_word2idx or tgtword not in tgt_word2idx:
            continue
        src_align_emb.append(src_emb_matrix[src_word2idx[srcword]])
        tgt_align_emb.append(tgt_emb_matrix[tgt_word2idx[tgtword]])

    src_align_matrix = torch.stack(src_align_emb,dim=0)
    tgt_align_matrix = torch.stack(tgt_align_emb,dim=0)
    sim_matrix = tgt_align_matrix.transpose(0,1) @ src_align_matrix
    svd_results = torch.svd(sim_matrix)
    U,V = svd_results.U, svd_results.V

    src_projected = src_emb_matrix @ V.transpose(0,1)
    tgt_projected = tgt_emb_matrix @ U.transpose(0,1)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path,trust_remote_code=True)

    embeddings = torch.randn((len(tokenizer),512)).cuda()
    for srcword in src_word2idx:
        emb = src_projected[src_word2idx[srcword]]
        if srcword in tgt_word2idx:
            tgt_emb = tgt_projected[tgt_word2idx[tgtword]]
            emb = (emb + tgt_emb) / 2
        idx = tokenizer.convert_tokens_to_ids(srcword)
        embeddings[idx] = emb

    for tgtword in tgt_word2idx:
        emb = tgt_projected[tgt_word2idx[tgtword]]
        if tgtword in src_word2idx:
            src_emb = src_projected[src_word2idx[tgtword]]
            emb = (emb + src_emb) / 2
        idx = tokenizer.convert_tokens_to_ids(tgtword)
        embeddings[idx] = emb

    evaluate_NN(embeddings,tokenizer,["Tim","play","garden"])

    
        


    import pdb; pdb.set_trace()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-emb")
    parser.add_argument("--tgt-emb")
    parser.add_argument("--dict-file")
    parser.add_argument("--tokenizer-path")

    args = parser.parse_args()
    main(args)
