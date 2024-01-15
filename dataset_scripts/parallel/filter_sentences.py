
from comet import download_model, load_from_checkpoint
import re
from tqdm import tqdm
import json
import argparse
import torch

CONTAINS_CHINESE = re.compile(r'[\u4e00-\u9fff]+')

def predict(datas,device):
    # Choose your model from Hugging Face Hub
    model_path = download_model("Unbabel/wmt22-cometkiwi-da")

    # Load the model checkpoint:
    model = load_from_checkpoint(model_path)

    # Call predict method:
    model_output = model.predict(datas, batch_size=512, devices=[device])

    with open("paracrawl.with_score_{}.json".format(device),"w") as fout:
        for d,s in sorted(zip(datas,model_output[0]),key=lambda x: x[1],reverse=True):
            fout.write(json.dumps(
                {
                    "src_text": d["src"],
                    "tgt_text": d["mt"],
                    "score": s
                }
            ) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile")
    parser.add_argument("--outfile")
    parser.add_argument("--devices")

    args = parser.parse_args()

    datas = []
    with open(args.infile) as f:
        for line in tqdm(f):
            english, chinese = tuple(line.split("\t"))
            if not CONTAINS_CHINESE.search(chinese):
                continue
            if CONTAINS_CHINESE.search(english):
                continue
            datas.append(
                {
                    "src": english,
                    "mt": chinese
                }
            )

    devices = [int(i) for i in args.devices.split(",")]

    chunk_size = len(datas) // len(devices)
    manager = torch.multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    for i, device_id in enumerate(devices):
        start, end = chunk_size*i, min(chunk_size*(i+1),len(datas))
        sub_datas = datas[start:end]
        p = torch.multiprocessing.Process(target=predict, args=(sub_datas,device_id))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()
