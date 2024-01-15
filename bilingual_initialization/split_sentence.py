
import sys
import json
from tqdm import tqdm
import re

infile, outfile = sys.argv[1], sys.argv[2]

class Spliter:
    def __init__(self,lang):
        self.lang = lang
        if lang == "en":
            self.pattern = re.compile(r"\.\"|\!|\.")
        else:
            self.pattern = re.compile(r"。”|\！|。")

    def split(self,document):
        return self.pattern.split(document)



en_spliter, zh_spliter = Spliter("en"), Spliter("zh")


with open(infile) as f:
    mismatched = 0
    all_data = set()
    for line in tqdm(f):
        d = json.loads(line)
        en_raw, zh_raw = d["src_text"], d["tgt_text"]
        en_cut_sentences, zh_cut_sentences = en_spliter.split(en_raw), zh_spliter.split(zh_raw)
        if len(en_cut_sentences) != len(zh_cut_sentences):
            mismatched += 1
        else:
            for en,zh in zip(en_cut_sentences,zh_cut_sentences):
                if len(en) == 0 and len(zh) == 0:
                    continue
                all_data.add((en,zh))

    with open(outfile,"w") as fout:
        for en,zh in all_data:
            fout.write(json.dumps(
                {"src_text":en, "tgt_text":zh}, ensure_ascii=False
            ) + "\n")

    print(mismatched)
