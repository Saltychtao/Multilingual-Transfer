import json
from tqdm import tqdm
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/mnt/bn/st-data-lq/jiahuanli/models/Baichuan2-13B-base",trust_remote_code=True)
total_length = 0
with open("/mnt/bn/st-data-lq/jiahuanli/data/paracrawl/paracrawl.json") as f, open("paracrawl.clean.json","w") as fout:
    datas = []
    for line in tqdm(f):
        d = json.loads(line)
        datas.append(d)
    prev = 0
    for d in sorted(datas,key=lambda x: x["score"],reverse=True):
        tokenized = tokenizer(d["src_text"]  + d["tgt_text"])
        total_length += len(tokenized["input_ids"])
        fout.write(json.dumps(d,ensure_ascii=False) + "\n")

        print(total_length)
        if total_length > 32000000:
            break
