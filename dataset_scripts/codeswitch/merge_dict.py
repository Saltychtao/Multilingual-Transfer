
with open("dict-all.txt","w") as fout, open("dict-full.txt") as f_full, open("extended_dict.txt") as f_extend :
    for line in f_full:
        src,tgt = tuple(line.split())
        fout.write(src + "\t" + tgt + "\n")
    for line in f_extend:
        fout.write(line)

with open("dict-full.txt") as f, open("dict-full2.txt","w") as fout:
    for line in f:
        src,tgt = tuple(line.split())
        fout.write(src + "\t" + tgt + "\n")
        