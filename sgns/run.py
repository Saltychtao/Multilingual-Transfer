from functools import partial
from transformers import AutoTokenizer
from datasets import load_dataset
import random
from itertools import chain
import torch
import transformers
from dataclasses import dataclass
import torch.nn.functional as F
import argparse
from tqdm import tqdm

from sgns.data_utils import build_data_module

class Model(torch.nn.Module):
    def __init__(self,vocab_size,hidden_dim, pad_token_id):
        super().__init__()
        self.vocab_size = vocab_size
        self.inner_embedding = torch.nn.Embedding(vocab_size,hidden_dim)
        self.inner_embedding.weight.data[pad_token_id,:] = 0
        torch.nn.init.normal_(self.inner_embedding.weight,0,0.02)

        self.outer_embedding = torch.nn.Embedding(vocab_size,hidden_dim)
        self.outer_embedding.weight.data[pad_token_id,:] = 0
        torch.nn.init.normal_(self.outer_embedding.weight,0,0.02)

    def inner_embed(self,x):
        return F.normalize(self.inner_embedding(x),dim=-1)

    def outer_embed(self,x):
        return F.normalize(self.outer_embedding(x),dim=-1)

    def lookup(self,x):
        # return 0.5*(self.inner_embed(x) + self.outer_embed(x))
        return self.inner_embed(x)

    def project_out(self,embedding):
        return torch.nn.functional.linear(F.normalize(embedding,dim=-1),F.normalize(self.embedding.weight,dim=-1))


def compute_loss(model,tokenizer,batch,window_size,tau):
    tokens = batch["input_ids"].cuda()
    tokens = tokens.view(-1) # V
    T,device = tokens.size(0), tokens.device
    inner_embeddings, outer_embeddings = model.inner_embed(tokens), model.outer_embed(tokens) # T,h
    scores = torch.einsum("ih,jh->ij",(inner_embeddings,outer_embeddings)) # T,T
    logits = scores.div(tau)
    mask = torch.zeros((T,T),device=device,dtype=torch.bool)
    for i in range(1,window_size+1):
        mask |= torch.diag(torch.ones((T-i,),dtype=torch.bool,device=device),i)
        mask |= torch.diag(torch.ones((T-i,),dtype=torch.bool,device=device),-i)

    logit_mask = ~tokens.unsqueeze(-1).eq(tokens.unsqueeze(0))
    mask = mask * logit_mask
    # logits = logits - logits.masked_fill(~logit_mask,-float("inf")).max(dim=-1,keepdim=True)[0].detach()
    
    exp_logits = torch.exp(logits) * logit_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
    loss = (mask * log_prob ).sum(1) / mask.sum(1).clamp(1)

    return -loss.mean()

def evaluate_NN(model,tokenizer,word_list):
    inputs = tokenizer(word_list,return_tensors="pt",padding=True)
    tokens = inputs["input_ids"].cuda()
    query_embedding = model.lookup(tokens)
    all_embedding = model.lookup(torch.arange(0,len(tokenizer)-1).to(tokens))
    dist = torch.cdist(query_embedding.unsqueeze(0),all_embedding.unsqueeze(0),p=2).squeeze() # B x B
    # dist = torch.einsum("ih,jh->ij",(query_embedding,all_embedding))
    for i, word in enumerate(word_list):
        nearest_ids = dist[i].topk(k=10,dim=-1,largest=False).indices
        print(word, tokenizer.convert_ids_to_tokens(nearest_ids))

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path,trust_remote_code=True)
    model = Model(len(tokenizer),args.embed_dim,tokenizer.pad_token_id).cuda()
    if args.finetune_from is not None:
        pretrained = torch.load(args.finetune_from)
        model.embedding.weight.data = pretrained

    frequencies = torch.tensor(torch.load(args.frequencies_file))
    frequencies = 1 - (args.subsampling_ratio / frequencies).sqrt()

    dataset,collator = build_data_module(args.datapath,tokenizer,frequencies,args)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=args.batch_size,collate_fn=collator,shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
    
    for epoch in range(args.epochs):
        losses = 0.
        for i,batch in tqdm(enumerate(dataloader)):
            loss = compute_loss(model,tokenizer,batch,window_size=args.window_size,tau=args.tau)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses += loss.item()

            if i > 0 and i % 100 == 0:
                evaluate_NN(model,tokenizer,word_list=args.evaluate_word_list.split())
                print("Training Loss at Epoch {}, step {}, loss {}".format(epoch,i,losses/100))
                losses = 0.

    torch.save(model.embedding.weight.data,args.save_path)
                
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath")
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--epochs",default=3,type=int)
    parser.add_argument("--embed-dim",default=512,type=int)
    parser.add_argument("--batch-size",type=int,default=32)
    parser.add_argument("--tau",type=float,default=0.1)
    parser.add_argument("--testset")
    parser.add_argument("--lr",type=float)
    parser.add_argument("--save-path")
    parser.add_argument("--finetune-from")
    parser.add_argument("--window-size",default=5,type=int)
    parser.add_argument("--subsampling-ratio",default=1e-8)
    parser.add_argument("--frequencies-file",)
    parser.add_argument("--evaluate-word-list",type=str,default="Tim play happy garden")


    args = parser.parse_args()
    main(args)