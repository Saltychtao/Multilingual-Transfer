from transformers.trainer_utils import seed_worker
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch
from typing import Dict
from collections import defaultdict
import math
import os
import sys

from accelerate.data_loader import DataLoaderStateMixin, GradientState, synchronize_rng_states, send_to_device, AcceleratorState

from src.trainer.trainer import (
    Trainer,
    RandomSampler, 
)
from src.trainer.trainer_utils import parse_layers, build_metric


class ParallelPretrainer(Trainer):

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
        
        self._total_loss_scalar_dict = defaultdict(lambda: 0)

    def compute_lm_loss(self,model,lm_inputs):
        
        output, _ = model(**lm_inputs)
        lm_loss = output["loss"]

        if self.codeswitch_ratio >= 0:
            codeswitch_lm_inputs = self.build_codeswitch_data(lm_inputs)
            codeswitch_output, _ = model(**codeswitch_lm_inputs)
            lm_codeswitch_loss = codeswitch_output["loss"]
            lm_loss = lm_codeswitch_loss + lm_loss
        return lm_loss

    def compute_loss(self, model, inputs, return_outputs=False):
        if not model.training:
            outputs, _ = model(**inputs)
            loss = outputs["loss"]
            return (loss, outputs) if return_outputs else loss
        loss = 0.

        return_dict = {}
        en_output, en_state_dict = model(**inputs["x"])
        fr_output, fr_state_dict = model(**inputs["y"])
        en_mask = inputs["parallel"]["x"]["attention_mask"]
        fr_mask = inputs["parallel"]["y"]["attention_mask"]
        layer_index = self.cons_layers.to(en_mask.device)
        inconsistencies = []
        for pattern, metric in zip(self.cons_patterns, self.cons_metrics):
            source_pattern, target_pattern = en_state_dict[pattern].index_select(index=layer_index,dim=0),fr_state_dict[pattern].index_select(index=layer_index,dim=0)
            # L,B,T,H
            inconsistencies.append(
                metric(
                    source_pattern,
                    target_pattern)
                )
        inconsistencies_loss = sum(inconsistencies)
        loss += self.cons_strength * inconsistencies_loss
        return_dict["state_alignment_loss"] = inconsistencies_loss
        return return_dict

    