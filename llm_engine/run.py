from tabnanny import verbose
import openai
import argparse
import json
import yaml
from tqdm import tqdm
import time

from glob import glob

import multiprocessing
import os

import random

from tasks import build_task

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    wait_random
)  # for exponential backoff

class UnrecoverableError(Exception): pass


openai.api_type = "azure"
openai.api_version = "2023-05-15"
openai.api_key = "WwfRULbMCY0Fo9HYZZq1kEDLT7C8jyEm"
openai.api_base = "https://search.bytedance.net/gpt/openapi/online/v2/crawl"


@retry(wait=wait_random_exponential(min=60, max=120), stop=stop_after_attempt(60))
def chat_completion(model, meta_prompt, prompt, temperature=0.2, top_p=0.9, max_tokens=4000):
    messages = [{"role":"system", "content": meta_prompt}, {"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        engine="gpt_openapi",
        model=model, #必填，可选：gpt-35-turbo、gpt-4、gpt-4-32k
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        messages=messages,
        request_timeout=120
    )
    
    resp_text = response['choices'][0]['message']['content'].strip()    
    return resp_text

def process_batch(task,model,batch):
    prompt = batch["prompt"]
    output = chat_completion(model,task.meta_prompt,prompt,temperature=task.temperature)
    parsed_data, status = task.parse_output(output,batch)
    if status:
        ret = []
        for pd in parsed_data:
            ret.append(pd)
    else:
        ret = [{"output": parsed_data}]
    return ret

def process_chunks(args,data,index,return_dict):
    time.sleep(index*random.randint(0,30))
    task = build_task(args.task)
    batches = task.construct_inputs(data,args.batch_size)
    datas = []
    with open(args.savefile+".part-{}.json".format(index),"w",buffering=1) as fout:
        for batch in tqdm(batches):
            if args.debug:
                ret = process_batch(task,args.model,batch)
                datas.extend(ret)
            else:
                try:
                    ret = process_batch(task,args.model,batch)
                except Exception as inst:
                    print(inst)
                    ret = []
                    continue
            datas.extend(ret)

            for d in ret:
                fout.write(json.dumps(d,ensure_ascii=False) + "\n")

    return_dict[index] = datas
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",choices=["gpt-35-turbo","gpt-4"])
    parser.add_argument("--data-file")
    parser.add_argument("--savefile")
    parser.add_argument("--batch-size",type=int,default=10)
    parser.add_argument("--num-threads",type=int,default=8)
    parser.add_argument("--subset-size",type=int,default=-1)
    parser.add_argument("--task",)
    parser.add_argument("--debug",default=False,action="store_true")
    args = parser.parse_args()


    with open(args.data_file) as f:
        data = [json.loads(line.strip()) for line in f.readlines()]
        # random.shuffle(data)
        data = data[:args.subset_size]

    if args.debug:
        data = data[:10]
        return_dict = {}
        process_chunks(args,data,0,return_dict)
        print(return_dict)
        with open("../tmp2.json","w") as fout:
            json.dump(return_dict,fout,indent=4,ensure_ascii=False)
    else:
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        jobs = []

        chunk_size = len(data) // args.num_threads
        for thread_num in range(args.num_threads):
            chunk = data[thread_num*chunk_size: min((thread_num+1)*chunk_size,len(data))]
            p = multiprocessing.Process(target=process_chunks, args=(args,chunk, thread_num,return_dict))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()

        final_data = []
        for data in return_dict.values():
            final_data += data
            
        print("Final Data", len(final_data))
        with open(args.savefile,"w") as fout:
            json.dump(final_data, fout, ensure_ascii=False, indent=4)


        for file in glob("{}.part*".format(args.savefile)):
            os.remove(file)

        