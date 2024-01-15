from ast import arg
from transformers import AutoTokenizer
from datasets import load_dataset
import datasets
import torch
import transformers
from functools import partial
from dataclasses import dataclass
import argparse
from functools import partial
from tqdm import tqdm
import json
import torch.nn.functional as F
from geomloss import SamplesLoss

@dataclass
class DataCollatorForParallelDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        entry = []
        for key in ("english_input_ids","chinese_input_ids"):
            if key not in instances[0]:
                continue
            entry.extend([torch.tensor(instance[key]).long() for instance in instances])
        data = torch.nn.utils.rnn.pad_sequence(
            entry, batch_first=True, padding_value=self.tokenizer.pad_token_id,)
        return_dict = {
            "input_ids": data,
            "attention_mask": data.ne(self.tokenizer.pad_token_id),
            "labels": data.masked_fill(data.eq(self.tokenizer.pad_token_id),-100)
        }
        return return_dict

def tokenize_parallel_function(tokenizer,examples):
    tokenized_english = tokenizer(examples["src_text"])
    tokenized_chinese = tokenizer(examples["tgt_text"])

    return {
        "english_input_ids": [s+ [tokenizer.eos_token_id] for s in tokenized_english["input_ids"]],
        "chinese_input_ids": [s+ [tokenizer.eos_token_id] for s in tokenized_chinese["input_ids"]],
    }


def build_dataset(file,tokenize_fn,group_fn=None):

    dataset_args = {}
    raw_datasets = load_dataset("json", data_files={"train":file},**dataset_args)
    column_names = raw_datasets["train"].column_names

    dataset = raw_datasets.map(
        tokenize_fn,
        batched=True,
        num_proc=128,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )
    if group_fn is not None:
        dataset = dataset.map(
            group_fn,
            batched=True,
            num_proc=120,
            load_from_cache_file=True,
            desc="Grouping texts in chunks of 1024"
        )
    return dataset["train"]


def build_data_module(datapath,tokenizer):
    tokenize_fn = partial(tokenize_parallel_function,tokenizer)
    dataset = build_dataset(datapath,tokenize_fn)
    data_collator = DataCollatorForParallelDataset(tokenizer=tokenizer)
    return dataset, data_collator


class Model(torch.nn.Module):
    def __init__(self,vocab_size,hidden_dim, pad_token_id):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = torch.nn.Embedding(vocab_size,hidden_dim)
        self.embedding.weight.data[pad_token_id,:] = 0
        torch.nn.init.normal_(self.embedding.weight,0,0.02)

    def look_up(self,x):
        return self.embedding(x)

    def project_out(self,embedding):
        return torch.nn.functional.linear(F.normalize(embedding,dim=-1),F.normalize(self.embedding.weight,dim=-1))


def compute_loss(model,tokenizer,tau,batch,idf):
    tokens,labels = batch["input_ids"].cuda(), batch["labels"].cuda()
    B,T = tokens.size()
    pad_masks = batch["attention_mask"].cuda()

    seq_length = pad_masks.sum(dim=-1,keepdim=True)

    #compute monolingual cbow loss
    cbow_mask = torch.ones(size=(T,))
    cbow_mask = cbow_mask.to(pad_masks).diag()
    cbow_tokens = tokens.unsqueeze(1).masked_fill(cbow_mask.unsqueeze(0),tokenizer.pad_token_id)

    cbow_embeddings = model.look_up(cbow_tokens).sum(dim=1)  / (seq_length.unsqueeze(1) - 1)# B,T, H
    cbow_logits = model.project_out(cbow_embeddings) # B,T, V
    cbow_loss = torch.nn.functional.cross_entropy(cbow_logits.reshape(B*T,-1),labels.reshape(B*T))
    # compute bilingual cbow loss
    en_tokens, zh_tokens = tokens.chunk(2,dim=0)
    en_seq_length, zh_seq_length = seq_length.chunk(2,dim=0)
    en_cbow_embeddings = model.look_up(en_tokens).sum(dim=1) / en_seq_length # B,H
    en_zh_scores = model.project_out(en_cbow_embeddings)


    zh_cbow_embeddings = model.look_up(zh_tokens).sum(dim=1) / zh_seq_length
    zh_en_scores = model.project_out(zh_cbow_embeddings) # B,V => T1 pos, T2 neutral

    en_mask = torch.scatter_add(torch.zeros((B//2,model.vocab_size),device=tokens.device,dtype=torch.long),dim=-1,index=en_tokens,src=torch.ones_like(en_tokens)).bool()
    zh_mask = torch.scatter_add(torch.zeros((B//2,model.vocab_size),device=tokens.device,dtype=torch.long),dim=-1,index=zh_tokens,src=torch.ones_like(zh_tokens)).bool()

    en_mask[:,tokenizer.pad_token_id] = 0
    zh_mask[:,tokenizer.pad_token_id] = 0


    zh_weighting = zh_mask * idf.unsqueeze(0)
    en_zh_logits = en_zh_scores.div(tau)
    en_zh_logits = en_zh_logits - en_zh_logits.max(dim=1,keepdim=True)[0]
    en_zh_all_logits = en_zh_logits.logsumexp(dim=-1,keepdim=True)
    en_zh_logprob = en_zh_logits - en_zh_all_logits
    en_zh_cbow_loss = (zh_weighting * en_zh_logprob ).sum(1) / zh_mask.sum(1)

    en_weighting = en_mask * idf.unsqueeze(0)
    zh_en_logits = zh_en_scores.div(tau)
    zh_en_logits = zh_en_logits - zh_en_logits.max(dim=1,keepdim=True)[0]
    zh_en_all_logits = zh_en_logits.logsumexp(dim=-1,keepdim=True)
    zh_en_logprob = zh_en_logits - zh_en_all_logits
    zh_en_cbow_loss = (en_weighting * zh_en_logprob).sum(1) / en_mask.sum(1)

    return cbow_loss + en_zh_cbow_loss.neg().mean() + zh_en_cbow_loss.neg().mean()


def compute_aux_loss(model,tokenizer,tau,batch):
    tokens,labels = batch["input_ids"].cuda(), batch["labels"].cuda()
    B,T = tokens.size()
    pad_masks = batch["attention_mask"].cuda()
    seq_length = pad_masks.sum(dim=-1,keepdim=True)

    en_tokens, zh_tokens = tokens.chunk(2,dim=0)
    en_seq_length, zh_seq_length = seq_length.chunk(2,dim=0)
    en_embeddings, zh_embeddings = model.look_up(en_tokens) / en_seq_length.unsqueeze(-1), model.look_up(zh_tokens) / zh_seq_length.unsqueeze(-1)

    S = torch.einsum("bih,bjh->bij",(en_embeddings,zh_embeddings))
    S_fwd = S.softmax(dim=-1)
    S_bwd = S.softmax(dim=-2)
    A = (S_fwd * S_bwd) > 0.001
    SO_loss = 0.5*(S_fwd / en_seq_length.unsqueeze(-1) + S_bwd / zh_seq_length.unsqueeze(-1)) * A
    CO_loss = torch.bmm(S_fwd.transpose(1,2),S_bwd).diagonal(dim1=-2, dim2=-1).sum(-1) 
    return - (SO_loss.mean() + CO_loss.mean())

def load_testset(filename):
    with open(filename) as f:
        return json.load(f)

def get_embeddings(model,tokenizer,words):
    inputs = tokenizer(words,return_tensors="pt",padding=True)
    tokens = inputs["input_ids"].cuda()
    pad_mask = inputs["attention_mask"].cuda()

    seq_length = pad_mask.sum(dim=-1,keepdim=True)
    embeddings = model.look_up(tokens).sum(dim=1) // seq_length
    return embeddings

def evaluate(model,tokenizer,testset):
    english_words = [d["en"] for d in testset]
    chinese_words = [d["zh"] for d in testset]
    en_embeddings = get_embeddings(model,tokenizer,english_words)
    zh_embeddings = get_embeddings(model,tokenizer,chinese_words)

    dist = torch.einsum("ih,jh->ij",(F.normalize(en_embeddings,dim=-1),F.normalize(zh_embeddings,dim=-1)))
    # dist = torch.cdist(en_embeddings.unsqueeze(0),zh_embeddings.unsqueeze(0),p=2).squeeze(0) # B x B
    correct = (dist.argmax(dim=-1) == torch.arange(len(english_words)).cuda()).sum().item()
    total = len(english_words)

    return correct / total


def compute_idf(dataloader,model):
    results = torch.zeros((model.vocab_size,),device="cuda",dtype=torch.float)
    total = 0
    for batch in tqdm(dataloader):
        tokens,labels = batch["input_ids"].cuda(), batch["labels"].cuda()
        B,T = tokens.size()
        df = torch.zeros((B,model.vocab_size,),device="cuda",dtype=torch.float)
        df.scatter_(1, tokens, torch.ones((B,T), dtype=torch.float).cuda())
        results += df.sum(dim=0)
        total += B
    idf = torch.log(torch.tensor(total).float() / results.float().clamp(min=1))
    return idf

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path,trust_remote_code=True)
    model = Model(len(tokenizer),args.embed_dim,tokenizer.pad_token_id).cuda()
    if args.finetune_from is not None:
        pretrained = torch.load(args.finetune_from)
        model.embedding.weight.data = pretrained

    dataset, collator = build_data_module(args.datapath,tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=args.batch_size,collate_fn=collator)
    testset = load_testset(args.testset)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
    
    idf = compute_idf(dataloader,model)
    for epoch in range(args.epochs):
        losses = 0.
        for i,batch in tqdm(enumerate(dataloader)):
            loss = compute_loss(model,tokenizer,args.tau,batch,idf)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses += loss.item()

            if i > 0 and i % 100 == 0:
                accuracy = evaluate(model,tokenizer,testset)
                print("Training Loss at Epoch {}, step {}, loss {}, acc {:.4f}".format(epoch,i,losses/100,accuracy))
                losses = 0.

    torch.save(model.embedding.weight.data,args.save_path)
                
                




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath")
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--epochs",default=3,type=int)
    parser.add_argument("--embed-dim",default=512,type=int)
    parser.add_argument("--batch-size",type=int,default=32)
    parser.add_argument("--tau",type=float,default=1)
    parser.add_argument("--testset")
    parser.add_argument("--lr",type=float)
    parser.add_argument("--save-path")
    parser.add_argument("--finetune-from")

    args = parser.parse_args()
    main(args)











    