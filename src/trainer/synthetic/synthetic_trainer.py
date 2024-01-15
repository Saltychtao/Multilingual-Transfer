from transformers.trainer_utils import seed_worker
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch
from typing import Dict
import random
import os


from src.trainer.trainer import (
    Trainer,
    is_datasets_available,
    SequentialSampler,
    LengthGroupedSampler,
    RandomSampler,
    has_length,
    Optional
)
import datasets
from src.trainer.trainer_utils import parse_layers, build_metric
from src.trainer.synthetic.synthetic_converter import build_converter

class SyntheticTrainer(Trainer):

    def __init__(self,args,**kwargs):
        super().__init__(args=args,**kwargs)

        cons_patterns = args.cons_patterns.split(",")
        cons_metrics = args.cons_metrics.split(",")
        if len(cons_patterns) != len(cons_metrics):
            raise ValueError("The number of consistency patterns should be equal to the consistency metrics.")
        self.cons_strength = args.cons_strength
        self.cons_layers = torch.tensor(parse_layers(self.model.num_layers(),args.cons_layers)).long()
        self.cons_patterns = cons_patterns
        self.cons_metrics = [build_metric(metric) for metric in cons_metrics]

        self.contrastive_tau = args.contrastive_tau
        self.contrastive_p = args.contrastive_p

        self.ratio = [float(t) for t in args.synthesis_ratio.split("-")]
        self.converter = build_converter(args.converter_name,len(self.tokenizer))

        self.codeswitch_ratio = args.codeswitch_ratio
        self.codeswitch_token = args.codeswitch_token

        self.codeswitch_label_prediction_mode = args.codeswitch_label_prediction_mode

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        super()._save(output_dir,state_dict)

        torch.save(self.codeswitch_token, os.path.join(output_dir, "codeswitch_token.pt"))

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
            )
        elif self.args.sorted_dataset:
            return SequentialSampler(self.train_dataset)

        else:
            return RandomSampler(self.train_dataset)



    def split(self,inputs):
        input_ids, attention_mask, labels = inputs["input_ids"], inputs["attention_mask"], inputs["labels"]
        B = input_ids.size(0)
        portion = [int(r*B) for r in self.ratio]
        en_input_ids, syn_input_ids, parallel_input_ids = input_ids[:portion[0]], input_ids[portion[0]:portion[0]+portion[1]], input_ids[-portion[2]:]
        en_attention_mask, syn_attention_mask, parallel_attention_mask = attention_mask[:portion[0]], attention_mask[portion[0]:portion[0]+portion[1]], attention_mask[-portion[2]:]
        en_labels, syn_labels, parallel_labels = labels[:portion[0]], labels[portion[0]:portion[0]+portion[1]], labels[-portion[2]:]
        en_inputs = {
            "input_ids": en_input_ids,
            "attention_mask": en_attention_mask,
            "labels": en_labels
        }
        syn_inputs = {
            "input_ids": syn_input_ids,
            "attention_mask": syn_attention_mask,
            "labels": syn_labels
        }
        parallel_inputs = {
            "input_ids": parallel_input_ids,
            "attention_mask": parallel_attention_mask,
            "labels": parallel_labels
        }
        return en_inputs, syn_inputs, parallel_inputs

    def build_codeswitch_data(self,lm_input):
        input_ids = lm_input["input_ids"] # B,T
        # mask = torch.rand(input_ids.size(),device=input_ids.device,dtype=torch.float).lt(self.codeswitch_ratio) & input_ids.lt(self.codeswitch_from_top_k)
        mask = input_ids.unsqueeze(-1).eq(self.codeswitch_token.to(input_ids.device)).sum(dim=-1).gt(0) & torch.rand(input_ids.size(),device=input_ids.device,dtype=torch.float).lt(self.codeswitch_ratio)
        codeswitched_input_ids = self.converter.forward(lm_input)["input_ids"]
        new_input_ids = torch.where(
            mask, codeswitched_input_ids, input_ids
        )
        new_attention_mask =  lm_input["attention_mask"]
        if self.codeswitch_label_prediction_mode == "no_predict":
            # condition 1: do not predict the switched label
            new_labels = torch.where(
                mask, -100, input_ids
            )
        elif self.codeswitch_label_prediction_mode == "codeswitch":
            # condition 2: predict the codeswitch label
            new_labels = torch.where(
                mask, codeswitched_input_ids, input_ids
            )
        elif self.codeswitch_label_prediction_mode == "original":
            # condition 3: predict the original label
            new_labels = torch.where(
                mask, input_ids, input_ids
            )
        elif self.codeswitch_label_prediction_mode == "mix":
            codeswitched_labels = torch.where(
                mask, codeswitched_input_ids, input_ids
            )
            original_labels = torch.where(
                mask, input_ids, input_ids
            )
            new_labels = torch.where(
                torch.rand(input_ids.size(),device=input_ids.device,dtype=torch.float).lt(0.5),
                original_labels,
                codeswitched_labels
            )

        elif self.codeswitch_label_prediction_mode == "double":
            codeswitched_labels = torch.where(
                mask, codeswitched_input_ids, input_ids
            )
            original_labels = torch.where(
                mask, input_ids, input_ids
            )
            new_input_ids = torch.cat((new_input_ids,input_ids),dim=0)
            new_attention_mask = torch.cat((new_attention_mask,new_attention_mask),dim=0)
            new_labels = torch.cat((codeswitched_labels,original_labels),dim=0)
        else:
            raise ValueError("No such codeswitch label prediction mode")
        return {
            "input_ids": new_input_ids,
            "attention_mask": new_attention_mask,
            "labels": new_labels
        }


    def compute_loss(self, model, inputs, return_outputs=False):
        if not model.training:
            outputs, _ = model(**inputs)
            loss = outputs["loss"]
            return (loss, outputs) if return_outputs else loss

        loss = 0.
        device = inputs["input_ids"].device
        en_inputs, syn_inputs, parallel_en_inputs = self.split(inputs)

        if self.ratio[0] > 0:
            if self.codeswitch_ratio > 0:
                en_inputs = self.build_codeswitch_data(en_inputs)

            en_total_tokens = en_inputs["labels"].ne(-100).sum()
            en_output, _ = model(**en_inputs)
            en_loss = en_output["loss"]
            loss += en_loss * en_total_tokens
        else:
            en_loss = torch.tensor(0.).to(device)
            en_total_tokens = 1

        if self.ratio[1] > 0:
            syn_inputs = self.converter.forward(syn_inputs)
            syn_total_tokens = syn_inputs["labels"].ne(-100).sum()
            syn_output, _ = model(**syn_inputs)
            syn_loss = syn_output["loss"]
            loss += syn_loss * syn_total_tokens
        else:
            syn_loss = torch.tensor(0.).to(device)
            syn_total_tokens = 1

        loss /= en_total_tokens + syn_total_tokens

        if self.ratio[2] > 0 and self.cons_strength > 0:
            parallel_syn_inputs = self.converter.forward(parallel_en_inputs)
            _, en_state_dict = model(**parallel_en_inputs)
            _, syn_state_dict = model(**parallel_syn_inputs)
            inconsistencies = []
            for pattern, metric in zip(self.cons_patterns, self.cons_metrics):
                indices = self.cons_layers.to(en_loss.device)
                source_pattern, target_pattern = en_state_dict[pattern].index_select(index=indices,dim=0), syn_state_dict[pattern].index_select(index=indices,dim=0)
                inconsistencies.append(
                    metric(
                        source_pattern,
                        target_pattern)
                    )
            inconsistencies_loss = sum(inconsistencies)

            loss += self.cons_strength * inconsistencies_loss
        else:
            inconsistencies_loss = torch.tensor(0.).to(loss.device)


        return {
                "loss": loss,
                "en_loss": en_loss,
                "syn_loss": syn_loss,
                "incons_loss": inconsistencies_loss
        }

    