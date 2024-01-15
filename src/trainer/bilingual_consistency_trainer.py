from transformers.trainer_utils import seed_worker
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch
from lightning.pytorch.utilities import CombinedLoader
import datasets

from src.trainer.trainer_with_parallel import TrainerWithParallel

def constrastive_metric(x,y):
    # x : layers, bsz, dim
    # y : layers, bsz, dim
    sim = torch.einsum("lbh,lbh->lbb",(x,y))
    scores = sim.div(0.1)

def build_metric(metric):
    if metric == "l2":
        return lambda x,y: F.mse_loss(x,y)
    elif metric == "l1":
        return lambda x,y: (x-y).abs().mean()
    elif metric == "constrastive":
        return 

def build_aggr(aggr):
    if aggr == "sum":
        return lambda x: torch.sum(x,dim=1)
    elif aggr == "mean":
        return lambda x: torch.mean(x,dim=1)
    elif aggr == "max":
        return lambda x: torch.max(x, dim=1)

class BilingualConsistencyTrainer(TrainerWithParallel):
    def __init__(self,args,**kwargs):
        super().__init__(args=args,**kwargs)

        if len(args.consistency_patterns) != len(args.consistency_metrics):
            raise ValueError("The number of consistency patterns should be equal to the consistency metrics.")
        self.consistency_strength = args.consistency_strength
        self.consistency_patterns = args.consistency_patterns
        self.consistency_layer_begin = args.consistency_layer_begin
        self.consistency_layer_end = args.consistency_layer_end
        self.consistency_metrics = [build_metric(metric) for metric in args.consistency_metrics]

    def compute_loss(self, model, inputs, return_outputs=False):
        combined_inputs = {
            "input_ids": torch.cat((inputs["lm"]["input_ids"],inputs["mt"]["input_ids"]),dim=0),
            "labels": torch.cat((inputs["lm"]["labels"],inputs["mt"]["labels"]),dim=0),
            "attention_mask": torch.cat((inputs["lm"]["attention_mask"],inputs["mt"]["attention_mask"]),dim=0)
        }
        outputs = model(**combined_inputs)
        lm_loss = outputs["loss"]

        return lm_loss

    def compute_loss(self, model, inputs, return_outputs=False):
        lm_loss = super().compute_loss(model,inputs,return_outputs=False)

        _, source_outputs = model(**inputs["source"])
        _, target_outputs = model(**inputs["target"])

        inconsistencies = []
        for pattern, metric in zip(self.consistency_patterns, self.consistency_metrics):
            source_pattern = source_outputs[pattern][self.consistency_layer_begin:self.consistency_layer_end+1]
            target_pattern = target_outputs[pattern][self.consistency_layer_begin:self.consistency_layer_end+1]
            inconsistencies.append(metric(source_pattern,target_pattern))
        inconsistencies_loss = sum(inconsistencies)
        return lm_loss + self.consistency_strength * inconsistencies_loss





        



