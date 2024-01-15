import torch
import argparse
from tqdm import tqdm

def read_vector(vector_path):
    word2idx = {}
    idx2word = []
    with open(vector_path) as f:
        lines = f.readlines()
        meta = lines[0]
        word_cnt, dim = tuple(meta.split())
        word_cnt,dim = int(word_cnt), int(dim)
        vectors = []
        for line in tqdm(lines[1:]):
            splited = line.split()
            if len(splited) != dim+1:
                continue
            word2idx[splited[0]] = len(word2idx)
            idx2word.append(splited[0])
            vectors.append(list(map(float,splited[1:])))
    return word2idx, idx2word, torch.tensor(vectors).cuda()

def evaluate_NN(vectors,word2idx,idx2word,word_list):
    import pdb; pdb.set_trace()
    query_idx = []
    for w in word_list:
        query_idx.append(word2idx[w])
    query_idx = torch.tensor(query_idx).long().cuda() # i
    query_embed = vectors[query_idx] # i,h
    dist = torch.einsum("ih,jh->ij",(torch.nn.functional.normalize(query_embed,dim=-1),torch.nn.functional.normalize(vectors,dim=-1)))
    for i, word in enumerate(word_list):
        nearest_ids = dist[i].topk(k=10,dim=-1,largest=False).indices
        for i in nearest_ids.tolist():
            print(word, idx2word[i],end=" ")
    
def main(args):
    word2idx,idx2word, vectors = read_vector(args.vector_file)
    evaluate_NN(vectors,word2idx,idx2word,args.word_list.split())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector-file")
    parser.add_argument("--word-list",default="Tim play park dust")

    args = parser.parse_args()
    main(args)






    