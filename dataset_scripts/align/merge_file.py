import sys
import json
from tqdm import tqdm

ret_data = []
for split in ["0000","0001","0002","0003"]:
    infile = sys.argv[1] + "." + split
    alignfile = sys.argv[2] + "." + split
    with open(infile) as f, open(alignfile) as f_align:
        for line, align_line in tqdm(zip(f,f_align)):
            src,tgt = tuple(line.split(" ||| "))
            align_src_idx, align_tgt_idx = [],[]
            for align in align_line.strip().split():
                src_idx, tgt_idx = tuple(align.split("-"))
                align_src_idx.append(int(src_idx))
                align_tgt_idx.append(int(tgt_idx))

            ret_data.append(
                {
                    "src_text": src,
                    "tgt_text": tgt,
                    "align_src_idx": align_src_idx,
                    "align_tgt_idx": align_tgt_idx
                }
            )

with open(sys.argv[3],"w") as fout:
    for d in tqdm(ret_data):
        fout.write(json.dumps(d,ensure_ascii=False) + "\n")
