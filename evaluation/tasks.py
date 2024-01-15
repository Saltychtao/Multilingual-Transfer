import re
class NLITask:
    def __init__(self):
        pass

    def construct(self,d,using_label=False):
        if using_label:
            return "Lily: {} Bob: {} Does Bob agree with Lily? Answer: {}".format(d["premises"],d["hypothesis"],"Yes." if d["label"] == "entailment" else "No.")
        else:
            return "Lily: {} Bob: {} Does Bob agree with Lily? Answer: ".format(d["premises"],d["hypothesis"],"Yes." if d["label"] == "entailment" else "No.")

    def construct_few_shot(self,demonstrations,d):
        return "\n".join([self.construct(_d,using_label=True) for _d in demonstrations]) + " " + self.construct(d)

    def extract_label(self,prompt,output):
        return output.replace(prompt,"")


def build_prompter(task):
    if task == "nli":
        return NLITask()
