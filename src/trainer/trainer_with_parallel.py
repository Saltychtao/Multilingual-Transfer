from lib2to3.pgen2 import token
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



class ConcatDataLoaderShard(DataLoaderStateMixin):
    def __init__(
        self,
        dataloader_dict,
        device=None,
        rng_types=None,
        synchronized_generator=None,
        skip_batches=0,
        _drop_last: bool = False,
    ):
        self.dataloader_dict = dataloader_dict
        self.device = device
        self.rng_types = rng_types
        self.synchronized_generator = synchronized_generator
        self.skip_batches = skip_batches
        self.gradient_state = GradientState()
        self._drop_last = _drop_last
        self.iteration = 0

    def __iter__(self):
        if self.rng_types is not None:
            synchronize_rng_states(self.rng_types, self.synchronized_generator)
        self.begin()

        self.set_epoch(self.iteration)
        dataloader_iters = {k: v.__iter__() for k,v in self.dataloader_dict.items()}
        # We iterate one batch ahead to check when we are at the end
        try:
            current_batches = {k:next(v) for k,v in dataloader_iters.items()}
        except StopIteration:
            yield

        batch_index = 0
        while True:
            try:
                # But we still move it to the device so it is done before `StopIteration` is reached
                if self.device is not None:
                    current_batches = {k: send_to_device(v, self.device) for k,v in current_batches.items()}
                next_batches = {k:next(v) for k,v in dataloader_iters.items()}
                if batch_index >= self.skip_batches:
                    yield current_batches
                batch_index += 1
                current_batches = next_batches
            except StopIteration:
                self.end_of_dataloader = True
                if batch_index >= self.skip_batches:
                    yield current_batches
                break

        self.iteration += 1
        self.end()

    def set_epoch(self, epoch: int):
        # In case it is manually passed in, the user can set it to what they like
        if self.iteration != epoch:
            self.iteration = epoch
        for dataloader in self.dataloader_dict.values():
            if hasattr(dataloader.batch_sampler, "sampler") and hasattr(dataloader.batch_sampler.sampler, "set_epoch"):
                dataloader.batch_sampler.sampler.set_epoch(epoch)
        # We support if a custom `Dataset` implementation has `set_epoch`
        # or in general HF datasets `Datasets`
            elif hasattr(dataloader.dataset, "set_epoch"):
                dataloader.dataset.set_epoch(epoch)

    @property
    def total_batch_size(self):
        raise NotImplementedError("total batch size not implemented")
        # batch_sampler = self.sampler if isinstance(self.sampler, BatchSampler) else self.batch_sampler
        # return (
        #     batch_sampler.batch_size
        #     if getattr(batch_sampler, "split_batches", False)
        #     else (batch_sampler.batch_size * getattr(batch_sampler, "num_processes", 1))
        # )

    @property
    def total_dataset_length(self):
        raise NotImplementedError("total dataset length not implemented")

    def __len__(self):
        return min([len(v) for v in self.dataloader_dict.values()])





class TrainerWithParallel(Trainer):

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

        self.ba_strength = args.ba_strength
        self.trans_strength = args.trans_strength

        self.sentswitch_alignment_strength = args.sentswitch_alignment_strength
        self.sentswitch_ratio = args.sentswitch_ratio

        self.codeswitch_corpus_ratio = args.codeswitch_corpus_ratio
        self.codeswitch_alignment_strength = args.codeswitch_alignment_strength

        self.codeswitch_alignment_type = "kl"
        
        self._total_loss_scalar_dict = defaultdict(lambda: 0)

    def prepare_dataloader(self,dataset,collator,batch_size):
        dataloader_params = {
            "batch_size": batch_size,
            "collate_fn": collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }
        if not isinstance(dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = RandomSampler(dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
        dataloader = self.accelerator.prepare(DataLoader(dataset,**dataloader_params))
        return dataloader


    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        lm_dataset, parallel_dataset = self.train_dataset["lm"],self.train_dataset["parallel"]
        lm_collator,  parallel_collator, _ = self.data_collator
    
        if parallel_dataset:
            mt_bsize = self._train_batch_size
            lm_dataloader = self.prepare_dataloader(lm_dataset,lm_collator,mt_bsize)

            parallel_dataloader = self.prepare_dataloader(
                parallel_dataset,
                parallel_collator,
                max(int(self._train_batch_size / (len(lm_dataset) / len(parallel_dataset))),1)
            )
            dataloader = ConcatDataLoaderShard(
                {
                    "lm": lm_dataloader,
                    "parallel": parallel_dataloader
                },
            )
            self.empty_parallel_dataset = False
        else:
            self.empty_parallel_dataset = True
            dataloader = self.prepare_dataloader(lm_dataset,lm_collator,self._train_batch_size)
        return dataloader

    def get_eval_dataloader(self, eval_dataset = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.data_collator[-1]

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last

        return self.accelerator.prepare(DataLoader(eval_dataset, **dataloader_params))


    def compute_ba_loss(self,xy_logits, yy_logits,length):
        # compute logit alignment
        device = xy_logits.device
        T1,T2 = xy_logits.size(1), yy_logits.size(1)
        xy_mask = torch.arange(0,T1).unsqueeze(0).to(device).gt(T1-length.unsqueeze(1))
        yy_mask = torch.arange(0,T2).unsqueeze(0).to(device).gt(T2-length.unsqueeze(1)) # B,T

        xy_logits = xy_logits[xy_mask] # N,V
        yy_logits = yy_logits[yy_mask] # N,V
        xy_logprob, yy_logprob = xy_logits.log_softmax(dim=-1), yy_logits.log_softmax(dim=-1)
        kld_loss = torch.nn.functional.kl_div(xy_logprob,yy_logprob,log_target=True,reduction="batchmean")
        return kld_loss

    def compute_lm_loss(self,model,lm_inputs):
        output, _ = model(**lm_inputs)
        lm_loss = output["loss"]
        return lm_loss, lm_inputs["attention_mask"].sum()

    def compute_codeswitch_lm_loss(self,model,inputs):

        if "align_mask" in inputs:
            align_mask = inputs.pop("align_mask")
            codeswitch_corpus_size = align_mask.size(0)
        else:
            align_mask = None
            codeswitch_corpus_size = 0
        output, _ = model(**inputs,output_hidden_states=True)
        lm_loss, lm_token = output.loss, inputs["attention_mask"].sum()

        if codeswitch_corpus_size > 0 and self.codeswitch_alignment_strength > 0:
            original_masks, codeswitch_masks = align_mask.chunk(2,dim=0)

            if self.codeswitch_alignment_type == "mse":
                hidden_states = torch.stack(output.hidden_states,dim=2) # B,T,L,H
                # compute alignment point score
                original_hidden_states, codeswitch_hidden_states = hidden_states[:codeswitch_corpus_size].chunk(2,dim=0)
                original_hidden_states = original_hidden_states[original_masks.nonzero(as_tuple=True)]
                codeswitch_hidden_states = codeswitch_hidden_states[codeswitch_masks.nonzero(as_tuple=True)]

                alignment_token = original_masks.sum()
                alignment_loss = torch.nn.functional.mse_loss(original_hidden_states.float(),codeswitch_hidden_states.float(),reduction="sum") / alignment_token
            elif self.codeswitch_alignment_type == "kl":
                original_logits, codeswitch_logits = output.logits.chunk(2,dim=0)
                original_logits, codeswitch_logits = original_logits[original_masks.nonzero(as_tuple=True)], codeswitch_logits[codeswitch_masks.nonzero(as_tuple=True)]
                alignment_loss = torch.nn.functional.kl_div(original_logits.log_softmax(dim=-1),codeswitch_logits.log_softmax(dim=-1),log_target=True,reduction="batchmean")
                alignment_token = original_masks.sum()
        else:
            alignment_loss = torch.tensor(0.).to(lm_loss)
            alignment_token = 0

        return lm_loss, lm_token, alignment_loss, alignment_token        


    def compute_sentswitch_loss(self,model,sentswitch):
        output, _ = model(**sentswitch["input"])
        lm_loss, lm_token = output.loss, sentswitch["input"]["attention_mask"].sum()

        original_logits, sentswitch_logits, _ = output.logits.chunk(3,dim=0)
        original_masks, sentswitch_masks = sentswitch["masks"].chunk(2,dim=0)
        original_logits = original_logits[original_masks.nonzero(as_tuple=True)]
        sentswitch_logits = sentswitch_logits[sentswitch_masks.nonzero(as_tuple=True)]

        alignment_loss = torch.nn.functional.kl_div(original_logits.log_softmax(dim=-1),sentswitch_logits.log_softmax(dim=-1),log_target=True,reduction="batchmean")
        alignment_token = original_masks.sum()
        return lm_loss, lm_token, alignment_loss, alignment_token

    def compute_loss(self, model, inputs, return_outputs=False):
        if not model.training:
            outputs, _ = model(**inputs)
            loss = outputs["loss"]
            return (loss, outputs) if return_outputs else loss
        if "lm" in inputs:
            lm_inputs = inputs["lm"]
        else:
            lm_inputs = inputs

        lm_loss, token_num, codeswitch_alignment_loss, codeswitch_aligment_token_num = self.compute_codeswitch_lm_loss(model,lm_inputs)
        return_dict = {
            "lm_loss": lm_loss,
            "codeswitch_alignment_loss": codeswitch_alignment_loss
        }
        if not self.empty_parallel_dataset:
            if self.sentswitch_ratio > 0:
                parallel_lm_loss, parallel_lm_token_num, sentswitch_alignment_loss, sentswitch_alignment_token_num = self.compute_sentswitch_loss(model,inputs["parallel"]["sentswitch"])
                return_dict["sentswitch_alignment_loss"] = sentswitch_alignment_loss
            else:
                parallel_lm_loss, parallel_lm_token_num = self.compute_lm_loss(model,inputs["parallel"]["x_and_y"])
                sentswitch_alignment_loss, sentswitch_alignment_token_num = 0,0

        else:
            parallel_lm_loss, parallel_lm_token_num = 0,0
            sentswitch_alignment_loss, sentswitch_alignment_token_num = 0,0

        loss = (
            lm_loss*token_num + self.codeswitch_alignment_strength*codeswitch_alignment_loss*codeswitch_aligment_token_num + \
                    parallel_lm_loss*parallel_lm_token_num + \
                            self.sentswitch_alignment_strength * sentswitch_alignment_loss * sentswitch_alignment_token_num) / \
            (token_num + codeswitch_aligment_token_num + parallel_lm_token_num +  sentswitch_alignment_token_num)

            # if self.trans_strength > 0 or self.ba_strength > 0:
            #     xy_output,_ = model(**inputs["parallel"]["xy"])
            #     yx_output,_ = model(**inputs["parallel"]["yx"])

            # if self.trans_strength > 0:
            #     translation_loss = 0.5 * (xy_output["loss"] + yx_output["loss"])
            #     loss += translation_loss
            #     return_dict["translation_loss"] = translation_loss

            # if self.cons_strength > 0:
            #     _, en_state_dict = model(**inputs["parallel"]["x"])
            #     _, fr_state_dict = model(**inputs["parallel"]["y"])
            #     en_mask = inputs["parallel"]["x"]["attention_mask"]
            #     fr_mask = inputs["parallel"]["y"]["attention_mask"]
            #     layer_index = self.cons_layers.to(lm_loss.device)
            #     inconsistencies = []
            #     for pattern, metric in zip(self.cons_patterns, self.cons_metrics):
            #         source_pattern, target_pattern = en_state_dict[pattern].index_select(index=layer_index,dim=0),fr_state_dict[pattern].index_select(index=layer_index,dim=0)
            #         source_pattern = source_pattern.sum(dim=2) / en_mask.sum(dim=-1).unsqueeze(0).unsqueeze(-1)
            #         target_pattern = target_pattern.sum(dim=2) / fr_mask.sum(dim=-1).unsqueeze(0).unsqueeze(-1)
            #         inconsistencies.append(
            #             metric(
            #                 source_pattern,
            #                 target_pattern)
            #             )
            #     inconsistencies_loss = sum(inconsistencies)
            #     loss += self.cons_strength * inconsistencies_loss
            #     return_dict["state_alignment_loss"] = inconsistencies_loss

            # if self.ba_strength > 0:
            #     yy_output, _ = model(**inputs["parallel"]["yy"])
            #     xy_ba_loss = self.compute_ba_loss(xy_output.logits,yy_output.logits,inputs["parallel"]["y_lengths"])

            #     xx_output, _ = model(**inputs["parallel"]["xx"])
            #     yx_ba_loss = self.compute_ba_loss(yx_output.logits,xx_output.logits,inputs["parallel"]["x_lengths"])

            #     ba_loss = 0.5 * (xy_ba_loss + yx_ba_loss)

            #     loss += self.ba_strength * ba_loss
            #     return_dict["behavior_alignment_loss"] = ba_loss

        return_dict["loss"] = loss
        return return_dict

    