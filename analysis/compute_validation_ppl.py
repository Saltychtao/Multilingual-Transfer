from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import torch
import argparse
from tqdm import tqdm



from src.trainer.synthetic.synthetic_converter import build_converter
def load_dataset(filename):
    texts = []
    with open(filename) as f:
        for line in f:
            texts.append(json.loads(line)["text"])
    return texts

def get_rank(x):
   vals = x[range(len(x)), range(len(x))]
   return (x > vals[:, None]).long().sum(1)

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path,trust_remote_code=True).cuda().eval()

    codeswitch_tokens = torch.load(args.codeswitch_token_file).cuda()
    dataset = load_dataset(args.valid_file)
    batch_size = 40

    converter = build_converter("one2one",len(tokenizer))

    similarity , id_sim, ood_sim = [], [],[]
    
    ranks = []
    id_ranks, ood_ranks = [],[]

    for i in tqdm(range(0,len(dataset),batch_size)):
        batch = dataset[i:min(i+batch_size,len(dataset))]
        inputs = tokenizer(batch,return_tensors="pt",padding=True).to("cuda")
        labels = inputs["input_ids"][:,1:]

        with torch.no_grad():
            output = model(**inputs,output_hidden_states=True)

            original_hidden_states = output.hidden_states[-1][:,:-1][labels.ne(0)]

            synthetic_inputs = converter.forward(inputs)
            synthetic_output = model(**synthetic_inputs,output_hidden_states=True)
            synthetic_hidden_states = synthetic_output.hidden_states[-1][:,:-1][labels.ne(0)]

            _sim = torch.nn.functional.cosine_similarity(original_hidden_states,synthetic_hidden_states,dim=-1) # B,

            _sim2 = torch.einsum(
                "ih,jh->ij", (torch.nn.functional.normalize(original_hidden_states,dim=-1), torch.nn.functional.normalize(synthetic_hidden_states,dim=-1))
            )

            flatten_labels = labels[labels.ne(0)]
            _rank = get_rank(_sim2)
            ranks.append(_rank)
            id_ranks.append(_rank[flatten_labels.unsqueeze(-1).eq(codeswitch_tokens).any(dim=-1)])
            ood_ranks.append(_rank[flatten_labels.unsqueeze(-1).ne(codeswitch_tokens).all(dim=-1)])

            similarity.append(_sim)
            id_sim.append(_sim[flatten_labels.unsqueeze(-1).eq(codeswitch_tokens).any(dim=-1)])
            ood_sim.append(_sim[flatten_labels.unsqueeze(-1).ne(codeswitch_tokens).all(dim=-1)])

    print(torch.cat(similarity,dim=-1).float().mean())
    print(torch.cat(id_sim,dim=-1).float().mean())
    print(torch.cat(ood_sim,dim=-1).float().mean())
    print(torch.cat(ranks,dim=-1).float().mean())
    print(torch.cat(id_ranks,dim=-1).float().mean())
    print(torch.cat(ood_ranks,dim=-1).float().mean())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-file")
    parser.add_argument("--model-path")
    parser.add_argument("--codeswitch-token-file")

    args = parser.parse_args()
    main(args)

        



