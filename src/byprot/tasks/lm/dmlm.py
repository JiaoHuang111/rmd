# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""
Task training logic for DMLM (Crystal Domain Masked Language Model).
The overall structure follows dplm.py; only the task registry name and the
model are swapped for DMLM.
"""

from typing import Any, Union

import torch
from lightning.pytorch.utilities import grad_norm
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F
from torchmetrics import MeanMetric, MinMetric

from byprot import utils
from byprot.tasks import TaskLitModule, register_task
from byprot.utils.config import compose_config as Cfg
import os
# get the logger handle
log = utils.get_logger(__name__)


def new_arange(x, *size):
    """Build an arange tensor on the same device as the input tensor x.
    Args:
        x: reference tensor, used to determine the device
        size: shape to generate; if empty, x.size() is used
    Returns:
        a Tensor of shape [*size] whose last dimension is an arange sequence
    """
    if len(size) == 0:
        size = x.size()
    return torch.arange(size[-1], device=x.device).expand(*size).contiguous()


# ----------------------------- #
# Task class: DMLMTrainingTask
# ----------------------------- #
@register_task("lm/dmlm")  # registers the task under the name lm/dmlm
class DMLMTrainingTask(TaskLitModule):
    """Lightning module wrapper for DMLM, used for training and evaluation."""

    # default config
    _DEFAULT_CFG: DictConfig = Cfg(
        learning=Cfg(
            noise="rdm",  # ['full_mask', 'random_mask']
            num_unroll=0,
            watch_t1_t2_loss=False,
            cal_constant_loss=False,
            weight="constant",
        ),
    )

    def __init__(
        self,
        model: Union[nn.Module, DictConfig],        # model config or instance
        criterion: Union[nn.Module, DictConfig],    # loss config or instance
        optimizer: DictConfig,                      # optimizer config
        lr_scheduler: DictConfig = None,            # LR scheduler config
        *,
        learning=_DEFAULT_CFG.learning,             # training-related config
    ):
        # if not hasattr(self.hparams, "model"):
        #     log.warning("[Warning] self.hparams.model not found! Using default model config.")
        #     self.hparams.model = {}
        log.info(f'Function lm/dmlm.py.__init__() Start!')
        super().__init__(model, criterion, optimizer, lr_scheduler)

        # save hyperparameters for logging and checkpoint resumption
        self.save_hyperparameters(logger=True)

        # build the model
        log.info(f'Function lm/dmlm.py.__init__() step 1: build model!')
        self.build_model()
        # grab the tokenizer (it lives inside the model)
        log.info(f'Function lm/dmlm.py.__init__() step 2: build tokenizer!')
        self.tokenizer = self.model.tokenizer
        self.loss_accumulator = []
        log.info(f'Function lm/dmlm.py.__init__() Done!')


    def setup(self, stage=None) -> None:
        """Per-stage initialization logic."""
        super().setup(stage)

        # build the loss function
        self.build_criterion()
        # build the evaluation metrics
        self.build_torchmetric()

        if self.stage == "fit":
            log.info(f"\n{self.model}")  # print the model architecture
        elif self.stage == "test":
            self.test_step_outputs = []  # stores test outputs

    def on_before_optimizer_step(self, optimizer):
        """Log gradient norms before the optimizer update."""
        if self.global_rank == 0:  # only on the main process
            grad_norm_dict = grad_norm(
                self.trainer.strategy.model, norm_type=2
            )
            self.log_dict(grad_norm_dict)

    def build_model(self):
        """Build the DMLM model from config."""
        log.info(f"Instantiating neural model <{self.hparams.model._target_}>")
        self.model = utils.instantiate_from_config(
            cfg=self.hparams.model, group="model"
        )

    def build_criterion(self):
        """Build the loss function from config and set its ignore_index."""
        self.criterion = utils.instantiate_from_config(
            cfg=self.hparams.criterion
        )
        # pad_token does not contribute to the loss
        self.criterion.ignore_index = self.tokenizer.pad_token_id

    def build_torchmetric(self):
        """Build the evaluation metrics (loss and ppl)."""
        self.eval_loss = MeanMetric()
        self.eval_nll_loss = MeanMetric()
        self.val_ppl_best = MinMetric()

    def step(self, batch):
        """One forward pass and loss computation.
        batch is a dict containing:
            - coords: [bsz, len, n_atoms, 3], atom coordinates
            - coord_mask: [bsz, len], mask of valid coordinates
            - lengths: [bsz, len], sequence lengths
            - tokens: [bsz, len], token sequence
        """
        weighting = self.hparams.learning.weight
        # CSP: datamodule(conditioning=composition) attaches an extra
        # condition_mask [B, L] to every batch (True = a composition condition
        # position, excluded from corruption and from the loss).
        # For the random-generation task (default) the batch has no such key ->
        # None, so the behaviour is unchanged.
        condition_mask = batch.get("condition_mask", None)
        # the model computes logits, target, loss_mask, weights
        logits, target, loss_mask, weights = self.model.compute_loss(
            batch, weighting=weighting, condition_mask=condition_mask
        )
        # todo: print logits and target and push them through the decoder to check for problems
        # print('Function Step: start decoding and saving to 0!')
        # self.test_decode_save(logits)
        # self.test_decode_save(target)
        # compute the loss with the criterion
        loss, logging_output = self.criterion(
            logits,
            target,
            loss_mask,
            weights,
            watch_t1_t2_loss=self.hparams.learning.watch_t1_t2_loss,
            cal_constant_loss=self.hparams.learning.cal_constant_loss,
        )

        # check for NaN so that training does not crash
        if torch.isnan(loss):
            print("Loss NAN on step ", self.global_step)
            loss = loss * 0
            logging_output["nll_loss"] = logging_output["nll_loss"] * 0
            logging_output["fullseq_loss"] = logging_output["fullseq_loss"] * 0
            logging_output["fullseq_nll_loss"] = (
                logging_output["fullseq_nll_loss"] * 0
            )
            logging_output["ppl"] = logging_output["ppl"] * 0
        # print(f"Total Loss: {loss.item():.6f}")
        return loss, logging_output

    def training_step(self, batch: Any, batch_idx: int):

        loss, logging_output = self.step(batch)
        # print(f"Step {self.global_step} OK!")
        # # print detailed loss information
        # if batch_idx % 100 == 0:
        #     print(f"\n=== Batch {batch_idx} (Step {self.global_step}) ===")
        #     print(f"Total Loss: {loss.item():.6f}")
        self.loss_accumulator.append(loss.item())
        # generate once every 1000 updates
        if self.global_step % 1000 == 0:
            avg_loss = sum(self.loss_accumulator) / len(self.loss_accumulator)
            print(f"\n=== Average loss over the last 1000 updates ===")
            print(f"Step {self.global_step}, average loss of the last 1000 steps: {avg_loss:.6f}")
            print(f"Number of samples: {len(self.loss_accumulator)}")
            print("=" * 40)
            self.test_generate(batch_idx)
            # reset the accumulator
            self.loss_accumulator = []

        # log the training metrics
        self.log("global_step", self.global_step, on_step=True, on_epoch=False, prog_bar=True)
        self.log("lr", self.lrate, on_step=True, on_epoch=False, prog_bar=True)

        for log_key in logging_output:
            log_value = logging_output[log_key]
            self.log(
                f"train/{log_key}",
                log_value,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )

        return {"loss": loss}

    # -------------------- #
    # validation and test logic
    # -------------------- #

    def test_decode_save(self, samples, batch_idx=-1):
        decoded_seqs = self.model.tokenizer.batch_decode(samples.tolist())
        # clean_seqs = [''.join(seq.split(' ')) for seq in decoded_seqs]

        # save to file
        filename = f"cif_results/generated_batch_-2.txt"
        os.makedirs(os.path.dirname(filename), exist_ok=True)  # hydra chdirs into the run dir, so create the directory ourselves
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(f"=== Sequences generated by batch {batch_idx} ===\n\n")
            for i, seq in enumerate(decoded_seqs):
                f.write(f"Sequence {i + 1} (length: {len(seq)}):\n")
                f.write(f"{seq}\n\n")

        print(f"Results saved to: {filename}")
        print(f"Number of generated samples: {len(decoded_seqs)}")

    def test_generate(self, batch_idx: int):
        device = self.device  # Lightning modules expose a device attribute

        input_tokens = torch.full((5, 400), 374, device=device, dtype=torch.long)

        # generate with dmlm
        samples = self.model.generate(
            input_tokens=input_tokens,
            max_iter=500,
        )
        self.test_decode_save(samples, batch_idx)
        # decode and print the results
        # decoded results

        # still print the first two samples to the console
        # for i, seq in enumerate(clean_seqs[:2]):
        #     print(f"Example sequence {i + 1}: {seq[:100]}...")

    def on_test_epoch_start(self) -> None:
        """At the start of testing, set the noise strategy to full_mask."""
        self.hparams.noise = "full_mask"

    def validation_step(self, batch: Any, batch_idx: int):
        """Single validation step."""
        import time
        from datetime import datetime

        # current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # print(f'[{current_time}] Start validation_step!')
        loss, logging_output = self.step(batch)

        # accumulate the evaluation metrics
        sample_size = logging_output["sample_size"]
        self.eval_loss.update(loss, weight=sample_size)
        self.eval_nll_loss.update(logging_output["nll_loss"], weight=sample_size)
        # print('Validation_step End!')
        return {"loss": loss}

    def on_validation_epoch_end(self):
        """Compute and log the final metrics at the end of validation or test."""
        log_key = "test" if self.stage == "test" else "val"

        # average loss and ppl over the whole validation set
        eval_loss = self.eval_loss.compute()
        self.eval_loss.reset()
        eval_nll_loss = self.eval_nll_loss.compute()
        self.eval_nll_loss.reset()
        eval_ppl = torch.exp(eval_nll_loss)

        # log the metrics
        self.log(f"{log_key}/loss", eval_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{log_key}/nll_loss", eval_nll_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{log_key}/ppl", eval_ppl, on_step=False, on_epoch=True, prog_bar=True)

        # during fitting, also update the best ppl
        if self.stage == "fit":
            self.val_ppl_best.update(eval_ppl)
            self.log(
                "val/ppl_best",
                self.val_ppl_best.compute(),
                on_epoch=True,
                prog_bar=True,
            )

        super().on_validation_epoch_end()
