import re


class NLITask:

    meta_prompt = """
Natural language inference is the task of determining whether a “hypothesis” is true (entailment), false (contradiction), or undetermined (neutral) given a “premise”.

I will give you a tiny story. Your goal is to generate ten examples for testing a model's ability of natural language inference. 

Requirement:
1. The topic and langauge style of the generated examples should be similar to sentences in the tiny story.
2. Contents between different examples should different
3. Balance the ratio between neutral, contradiction, and entailment


Your output format should be:

[Example 1]
premises: ... (a natural language sentence.)
hypothesis: ... (another sentence similar to premises)
label: ... (Entailment / Contradiction / Neutral)

...


Now, please process the following input:

"""

    temperature = 1.0

    def construct_inputs(self,dataset,batch_size):
        batches = []
        i = 0
        prompts = []
        while i < len(dataset):
            d = dataset[i]
            prompts.append("Tiny Story:{} \n Your Output:".format(d["text"]))
            i += 1
            if len(prompts) >= batch_size:
                batches.append({"prompt":"\n\n".join(prompts),"N":10,"datas":[]})
                prompts = []
        return batches

    def parse_output(self,output,batch):
        N = batch["N"]
        pattern = r'\[Example\s*\d+\]'
        splited = re.split(pattern,output)
        if splited[0] == "" and len(splited) == N+1:
            splited = splited[1:]
            rets = []
            for assessment in splited:
                rets.append({"synthesized_case": assessment.strip()})
            return rets, True
        else:
            print("Parse Error!")
            return [output], False
    