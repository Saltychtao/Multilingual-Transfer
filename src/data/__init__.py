from src.data.sft_dataset import make_sft_data_module
from src.data.seq2seq_dataset import make_seq2seq_data_module
from src.data.lm_dataset import make_lm_data_module
from src.data.lm_with_parallel_dataset import make_lm_with_parallel_module
from src.data.lm_with_synthesis_dataset import make_lm_with_synthesis_module

def make_data_module(data_args,model_args,tokenizer,type):
    if type == "sft":
        return make_sft_data_module(data_args,model_args,tokenizer)
    elif type == "seq2seq":
        return make_seq2seq_data_module(data_args,model_args,tokenizer)
    elif type == "lm":
        return make_lm_data_module(data_args,model_args,tokenizer)
    elif type == "lm_with_parallel":
        return make_lm_with_parallel_module(data_args,model_args,tokenizer)
    elif type == "lm_with_synthesis":
        return make_lm_with_synthesis_module(data_args,model_args,tokenizer)