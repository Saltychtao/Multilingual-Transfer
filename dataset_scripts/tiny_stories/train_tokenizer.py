import sentencepiece as spm
import sys
import json
from tqdm import tqdm
with open(sys.argv[1]) as f, open(sys.argv[2],"w") as fout:
    for line in tqdm(f):
        d = json.loads(line)
        fout.write(d["src_text"]+ "\n")
        fout.write(d["tgt_text"] + "\n")

# spm.SentencePieceTrainer.train(input=sys.argv[2], model_prefix='m', vocab_size=16000,model_type="bpe",)