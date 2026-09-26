# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


import os
from typing import List, Optional

import hydra
from lightning.pytorch.strategies import FSDPStrategy
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import Logger as LightopenningLoggerBase
import torch
from torch import nn

from byprot import utils

log = utils.get_logger(__name__)


def train(config: DictConfig) -> Optional[float]:
    """Contains the training pipeline. Can additionally evaluate model on a
    testset, using best weights achieved during training.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """
    print('Function <train> start!')
    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    # Convert relative ckpt path to absolute path if necessary
    ckpt_path = not config.train.get(
        "force_restart", False
    ) and config.train.get("ckpt_path")
    if ckpt_path:
        # convert a relative path into an absolute path

        ckpt_path = utils.resolve_ckpt_path(
            ckpt_dir=config.paths.ckpt_dir, ckpt_path=ckpt_path
        )
        if os.path.exists(ckpt_path):
            log.info(f"Resuming checkpoint from <{ckpt_path}>")
        else:
            log.info(
                f"Failed to resume checkpoint from <{ckpt_path}>: file not exists. Skip."
            )
            ckpt_path = None
        # ignore a missing checkpoint file

    # loading pipeline
    # load datamodule, task module (pl_module), loggers and callbacks
    log.info(f'Function utils.common_pipeline Start!')
    datamodule, pl_module, logger, callbacks = utils.common_pipeline(config)

    # ----------------------------------------------------------------------
    # Optional: weights-only pretrained initialization.
    # Strictly distinct from the resume above (train.ckpt_path):
    #   - resume: model + optimizer + scheduler + global_step all continue;
    #   - init_ckpt: only the model weights of a PL checkpoint are loaded into
    #     the freshly built pl_module; optimizer / lr_scheduler / global_step
    #     all start at 0 for this experiment (this runs before Lightning
    #     creates the optimizer).
    # Usage: train.init_ckpt=/path/to/last.ckpt (null by default in configs/config.yaml)
    # ----------------------------------------------------------------------
    init_ckpt = config.train.get("init_ckpt", None)
    if init_ckpt:
        init_ckpt = utils.resolve_ckpt_path(
            ckpt_dir=config.paths.ckpt_dir, ckpt_path=init_ckpt
        )
        if not os.path.exists(init_ckpt):
            raise FileNotFoundError(
                f"[init_ckpt] weights-only initialization checkpoint not found: {init_ckpt}"
            )
        _ckpt_sd = torch.load(init_ckpt, map_location="cpu")["state_dict"]
        missing, unexpected = pl_module.load_state_dict(_ckpt_sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"[init_ckpt] weights-only load failed: {len(missing)} missing / "
                f"{len(unexpected)} unexpected keys (see the lists above). "
                f"If the architectures do not match, drop train.init_ckpt and train from scratch."
            )
        log.info(
            f"Pretrained weights-only init from <{init_ckpt}>: "
            f"0 missing / 0 unexpected. optimizer/scheduler/global_step start from scratch (not a resume)."
        )

    # instantiate the PyTorch Lightning Trainer from the Hydra config
    # Init lightning trainer
    print('Init lightning trainer')
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    # write the hyperparameters to the loggers
    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=config,
        datamodule=datamodule,
        # model=model,
        model=pl_module,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # start training, resuming from a checkpoint if one is given
    # Train the model
    if config.get("train"):
        log.info("Starting training!")
        trainer.fit(
            model=pl_module, datamodule=datamodule, ckpt_path=ckpt_path
        )

    # read the requested optimized metric (e.g. validation loss or accuracy)
    # used for hyperparameter search
    # Get metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric and optimized_metric not in trainer.callback_metrics:
        raise Exception(
            "Metric for hyperparameter optimization not found! "
            "Make sure the `optimized_metric` in `hparams_search` config is correct!"
        )
    score = trainer.callback_metrics.get(optimized_metric)
    # evaluate on the test set with the best checkpoint

    # Test the model
    if config.get("test"):
        log.info("Starting testing!")
        best_ckpt_path = os.path.join(config.paths.ckpt_dir, "best.ckpt")
        trainer.test(
            model=pl_module, datamodule=datamodule, ckpt_path=best_ckpt_path
        )
    # training is over: release resources and close the loggers

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        model=pl_module,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Print path to best checkpoint
    if not config.trainer.get("fast_dev_run") and config.get("train"):
        log.info(
            f"Best model ckpt at {trainer.checkpoint_callback.best_model_path}"
        )
        # report the checkpoint path of the best model

    # Return metric score for hyperparameter optimization
    return score
