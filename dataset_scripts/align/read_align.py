import sys

datas = []
with open(sys.argv[1]) as f:
    for line in f:
        if len(datas) > 10:
            break
        src,tgt = tuple(line.strip().split(" ||| "))
        datas.append((src,tgt))
aligns = []
with open(sys.argv[2]) as f:
    for line in f:
        if len(aligns) > 10:
            break
        cur_align = []
        for a in line.split():
            a_s,a_t = tuple(a.split("-"))
            a_s,a_t = int(a_s), int(a_t)
            cur_align.append((a_s,a_t))
        aligns.append(cur_align)

for pair, align in zip(datas[:1],aligns[:1]):
    print(pair[0])
    src_words = pair[0].split()
    tgt_words = pair[1].split()
    for a_s,a_t in sorted(align,key=lambda x: x[0]):
        print(src_words[a_s],tgt_words[a_t])


