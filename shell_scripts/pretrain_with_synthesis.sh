set -e

train_file=${train_file:-""}
validation_file=${validation_file:-"/mnt/bn/st-data-lq/jiahuanli/data/minipile/data/valid.json"}

hf_config=${hf_config:-"/mnt/bn/st-data-lq/jiahuanli/models/llama-2-7b-hf"}
config_file=${config_file:-""}

tokenizer_path=${tokenizer_path:-""}

master_port=${master_port:-"2222"}
expr=${expr_name:-""}
project=${project_name:-""}

devices=${devices:-""}
ds_config=${ds_config:-"configs/deepspeed/stage1.json"}

output_dir=${output_dir:-""}

cons_layers=${cons_layers:-"all"}
cons_strength=${cons_strength:-"0"}
cons_patterns=${cons_patterns:-"resid_post"}
cons_metrics=${cons_metrics:-"l2"}

contrastive_tau=${contrastive_tau:-"0.1"}
contrastive_p=${contrastive_p:-"2"}

converter_name=${converter_name:-"one2one"}
synthesis_ratio=${synthesis_ratio:-"0.95-0.05-0.05"}

codeswitch_ratio=${codeswitch_ratio:-"0"}
codeswitch_dict_tokens=${codeswitch_dict_ratio:-"0"}
codeswitch_label_prediction_mode=${codeswitch_label_prediction_mode:-"codeswitch"}

bind_embedding=${bind_embedding:-"0"}
bind_embedding_mode=${bind_embedding_mode:-"all"}

pretrained_word_embedding_en=${pretrained_word_embedding_en:-""}
pretrained_word_embedding_syn=${pretrained_word_embedding_syn:-""}

sort_by_freq_score=${sort_by_freq_score:-"0"}

while [ $# -gt 0 ]; do
    if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
        # echo $1 $2 // Optional to see the parameter:value result
    fi
    shift
done

if [[ $pretrained_word_embedding_en != "" ]];then
    word_embedding_args="--pretrained_word_embedding_en ${pretrained_word_embedding_en}"
else
    word_embedding_args=""
fi

WANDB_PROJECT=${project} WANDB_NAME=${expr} deepspeed --master_port $master_port --include="localhost:$devices" src/pretrain_with_synthesis.py \
    --config_file $config_file \
    --train_file $train_file --validation_file $validation_file\
    --fp16 --do_train --do_eval\
    --hf_config $hf_config \
    --output_dir $output_dir \
    --seed 42 --report_to wandb \
    --tokenizer_path $tokenizer_path \
    --cons_layers $cons_layers --cons_metrics $cons_metrics \
    --cons_patterns $cons_patterns --cons_strength $cons_strength \
    --deepspeed $ds_config \
    --contrastive_tau $contrastive_tau --contrastive_p $contrastive_p \
    --converter_name $converter_name --synthesis_ratio $synthesis_ratio \
    --bind_embedding $bind_embedding \
    --codeswitch_ratio $codeswitch_ratio --codeswitch_dict_tokens $codeswitch_dict_tokens --codeswitch_label_prediction_mode $codeswitch_label_prediction_mode \
    $word_embedding_args --bind_embedding_mode $bind_embedding_mode \
    --sort_by_freq_score $sort_by_freq_score