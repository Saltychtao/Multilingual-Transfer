from transformers import AutoTokenizer
from datasets import load_dataset
import torch
import transformers
from functools import partial
from dataclasses import dataclass
import argparse
from functools import partial
from tqdm import tqdm
import json
import torch.nn.functional as F


@dataclass
class DataCollatorForParallelDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        return_dict = {}
        for key in ("english_input_ids","french_input_ids","alignment_src_idx","alignment_tgt_idx"):
            if key not in instances[0]:
                continue
            entry = [torch.tensor(instance[key]).long() for instance in instances]
            data = torch.nn.utils.rnn.pad_sequence(
                entry, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            return_dict[key]= data
        return return_dict

def tokenize_parallel_function(tokenizer,examples):
    tokenized_english = tokenizer(examples["src_text"])
    tokenized_french = tokenizer(examples["tgt_text"])

    return {
        "english_input_ids": tokenized_english["input_ids"],
        "french_input_ids": tokenized_french["input_ids"],
        "align_src_idx":  examples["align_src_idx"],
        "align_tgt_idx": examples["align_tgt_idx"]
    }


def build_dataset(file,tokenize_fn,group_fn=None):

    dataset_args = {}
    raw_datasets = load_dataset("json", data_files={"train":file},streaming=True,**dataset_args)
    column_names = raw_datasets["train"].column_names

    dataset = raw_datasets.map(
        tokenize_fn,
        batched=False,
        # num_proc=120,
        remove_columns=column_names,
        # load_from_cache_file=True,
        # desc="Running tokenizer on dataset",
    )
    if group_fn is not None:
        dataset = dataset.map(
            group_fn,
            batched=True,
            num_proc=120,
            # load_from_cache_file=True,
            # desc="Grouping texts in chunks of 1024"
        )
    return dataset["train"]


def build_data_module(datapath,tokenizer):
    tokenize_fn = partial(tokenize_parallel_function,tokenizer)
    dataset = build_dataset(datapath,tokenize_fn)
    data_collator = DataCollatorForParallelDataset(tokenizer=tokenizer)
    return dataset, data_collator


class Model(torch.nn.Module):
    def __init__(self,hidden_dim,):
        super().__init__()
        self.linear_x = torch.nn.Linear(hidden_dim,hidden_dim)
        self.linear_y = torch.nn.Linear(hidden_dim,hidden_dim)

    def project_x(self,x):
        return self.linear_x(x)

    def project_y(self,y):
        return self.linear_y(y)

def compute_loss(model,tokenizer,batch,src_emb, src_word2idx, tgt_emb,tgt_word2idx):
    src_tokens,tgt_tokens = batch["english_input_ids"].cuda(), batch["french_input_ids"].cuda()
    align_src_idx, align_tgt_idx = batch["align_src_idx"].cuda(), batch["align_tgt_idx"].cuda()
    

    return src_cbow_loss + tgt_cbow_loss

def load_testset(filename):
    with open(filename) as f:
        return json.load(f)

def get_embeddings(model,tokenizer,words):
    import pdb; pdb.set_trace()
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

def evaluate_NN(model,tokenizer,word_list):
    inputs = tokenizer(word_list,return_tensors="pt",padding=True)
    tokens = inputs["input_ids"].cuda()
    query_embedding = model.look_up(tokens)
    all_embedding = model.look_up(torch.arange(0,len(tokenizer)-1).to(tokens))
    dist = torch.cdist(query_embedding.unsqueeze(0),all_embedding.unsqueeze(0),p=2).squeeze() # B x B
    for i, word in enumerate(word_list):
        nearest_ids = dist[i].topk(k=5,dim=-1,largest=False).indices
        print(word, tokenizer.convert_ids_to_tokens(nearest_ids))

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

    evaluate_NN(model,tokenizer,args.evaluate_word_list.split())
    
    for epoch in range(args.epochs):
        losses = 0.
        for i,batch in tqdm(enumerate(dataloader)):
            loss = compute_loss(model,tokenizer,args.tau,batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses += loss.item()

            if i > 0 and i % 100 == 0:
                evaluate_NN(model,tokenizer,args.evaluate_word_list.split())
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
    parser.add_argument("--tau",type=float,default=1)
    parser.add_argument("--testset")
    parser.add_argument("--lr",type=float,default=1e-3)
    parser.add_argument("--save-path")
    parser.add_argument("--finetune-from")
    parser.add_argument("--evaluate-word-list",type=str,default="Tim play happy garden")

    args = parser.parse_args()
    main(args)











    