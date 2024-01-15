from datasets import load_dataset, concatenate_datasets
from glob import glob
import re
import argparse
from tqdm import tqdm

CONTAINS_CHINESE = re.compile(r'[\u4e00-\u9fff]+')

def main(args):
    pure_datasets, codeswitch_datasets = [],[]
    for train_file in tqdm(args.train_files):
        dataset = load_dataset("parquet",data_files={"train":train_file})["train"]

        pure = dataset.filter(lambda example: not CONTAINS_CHINESE.search(example["text"]) ,keep_in_memory=True,num_proc=128)
        codeswitch = dataset.filter(lambda example: CONTAINS_CHINESE.search(example["text"]) ,keep_in_memory=True,num_proc=128)

        pure_datasets.append(pure)
        codeswitch_datasets.append(codeswitch)

    pure_datasets = concatenate_datasets(pure_datasets)
    codeswitch_datasets = concatenate_datasets(codeswitch_datasets)

    pure_datasets.to_json(args.savedir+"/pure.json")
    codeswitch_datasets.to_json(args.savedir+"/codeswitch.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-files",nargs="+")
    parser.add_argument("--savedir")

    args = parser.parse_args()
    main(args)







