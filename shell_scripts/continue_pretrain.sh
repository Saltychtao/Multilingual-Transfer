set -e

train_file=${train_file:-""}
validation_file=${validation_file:-""}
output_dir=${output_dir:-""}

model_path=${model_type:-""}
config_file=${config_file:-""}

tokenizer_path=${tokenizer_path:-"/mnt/bn/st-data-lq/jiahuanli/models/pythia-70m"}

master_port=${master_port:-"2222"}
expr=${expr_name:-""}
project=${project_name:-"TinyStories"}

devices=${devices:-""}
ds_config=${ds_config:-"configs/stage1.json"}


while [ $# -gt 0 ]; do
    if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
        # echo $1 $2 // Optional to see the parameter:value result
    fi
    shift
done


WANDB_PROJECT=${project} WANDB_NAME=${expr} deepspeed --master_port $master_port --include="localhost:$devices" src/pretrain.py \
    --config_file $config_file \
    --train_file $train_file \
    --validation_file $validation_file \
    --fp16 --do_train \
    --model_name_or_path $model_path \
    --output_dir $output_dir \
    --seed 42 --report_to wandb  \
    --tokenizer_path $tokenizer_path \
    --do_eval 
