set -e

lm_train_file=${lm_train_file:-""}
parallel_train_file=${parallel_train_file:-""}
validation_file=${validation_file:-""}
output_dir=${output_dir:-""}

hf_config=${hf_config:-"/mnt/bn/st-data-lq/jiahuanli/models/llama-2-7b-hf"}
config_file=${config_file:-""}

tokenizer_path=${tokenizer_path:-""}

master_port=${master_port:-"2222"}
expr=${expr:-""}
project=${project:-""}

devices=${devices:-""}
ds_config=${ds_config:-"configs/deepspeed/stage1.json"}

cons_layers=${cons_layers:-"all"}
cons_strength=${cons_strength:-"0"}
cons_patterns=${cons_patterns:-"resid_post"}
cons_metrics=${cons_metrics:-"l2"}

bind_embeddings=${bind_embeddings:-"0"}
dict_file=${dict_file:-""}

trans_strength=${trans_strength:-"0"}
ba_strength=${ba_strength:-"0"}

codeswitch_ratio=${codeswitch_ratio:-"0"}
codeswitch_corpus_ratio=${codeswitch_corpus_ratio:-"0"}
sentswitch_ratio=${sentswitch_ratio:-"0"}
sentswitch_alignment_strength=${sentswitch_alignment_strength:-"0"}
codeswitch_alignment_strength=${codeswitch_alignment_strength:-"0"}

while [ $# -gt 0 ]; do
    if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
        # echo $1 $2 // Optional to see the parameter:value result
    fi
    shift
done


if [[ $dict_file != "" ]]; then
    dict_file_args="--dict_file ${dict_file}"
fi

WANDB_PROJECT=${project} WANDB_NAME=${expr} deepspeed --master_port $master_port --include="localhost:$devices" src/pretrain_with_parallel.py \
    --config_file $config_file \
    --lm_train_file $lm_train_file --parallel_train_file $parallel_train_file --validation_file $validation_file\
    --fp16 --do_train --do_eval --dataloader_num_workers 0\
    --hf_config $hf_config \
    --output_dir $output_dir \
    --seed 42 --report_to wandb \
    --tokenizer_path $tokenizer_path \
    --cons_layers $cons_layers --cons_metrics $cons_metrics \
    --cons_patterns $cons_patterns --cons_strength $cons_strength \
    --deepspeed $ds_config \
    $dict_file_args --bind_embeddings $bind_embeddings \
    --trans_strength $trans_strength --ba_strength $ba_strength \
    --codeswitch_ratio $codeswitch_ratio --codeswitch_corpus_ratio $codeswitch_corpus_ratio --codeswitch_alignment_strength $codeswitch_alignment_strength\
    --sentswitch_ratio $sentswitch_ratio --sentswitch_alignment_strength $sentswitch_alignment_strength