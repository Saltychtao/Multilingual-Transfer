import torch
class One2OneConverter:
    def __init__(self,vocab_size):
        self.vocab_size = vocab_size

    def forward(self,inputs):
        if "labels" not in inputs:
            return {
                "input_ids": torch.where(inputs["attention_mask"].ne(0), inputs["input_ids"] + self.vocab_size,  inputs["input_ids"]),
                "attention_mask": inputs["attention_mask"],
            }
        else:
            return {
                "input_ids": torch.where(inputs["attention_mask"].ne(0), inputs["input_ids"] + self.vocab_size,  inputs["input_ids"]),
                "attention_mask": inputs["attention_mask"],
                "labels": torch.where(inputs["attention_mask"].ne(0), inputs["labels"] + self.vocab_size,  inputs["labels"]),
            }

    def convert_list(self,inputs):
        return {
            "input_ids": [[x + self.vocab_size for x in xs] for xs in inputs["input_ids"]],
            "attention_mask": inputs["attention_mask"],
        }

def reverse_k_group(lst, k):
    n = len(lst)
    for i in range(0, n, k):
        lst[i:i+k] = lst[i:i+k][::-1]
    return lst

def reverse_k_group_torch(tensor, k):
    n, m = tensor.shape
    tensor = tensor.view(n, -1, k).permute(0, 2, 1).contiguous().view(n, -1)
    return tensor

class One2OneReorderConverter:
    def __init__(self,vocab_size):
        self.vocab_size = vocab_size
        self.k = 5
        self.reorder_indices = torch.tensor(reverse_k_group(list(range(512)),self.k)).long()

    def forward(self,inputs):

        return {
            "input_ids": (inputs["input_ids"] + self.vocab_size)[:,self.reorder_indices.to(inputs["input_ids"])],
            "attention_mask": inputs["attention_mask"],
            "labels": (inputs["labels"] + self.vocab_size)[:,self.reorder_indices.to(inputs["input_ids"])]
        }

    def convert_list(self,inputs):
        return {
            "input_ids": [reverse_k_group([x + self.vocab_size for x in xs],self.k) for xs in inputs["input_ids"]],
            "attention_mask": inputs["attention_mask"],
        }


def build_converter(converter_name,vocab_size):
    if converter_name == "one2one":
        return One2OneConverter(vocab_size)
    elif converter_name == "one2one_reorder":
        return One2OneReorderConverter(vocab_size)
