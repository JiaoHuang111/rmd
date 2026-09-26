# Copyright (c) 2025
# SPDX-License-Identifier: Apache-2.0
#
# DMLM: Diffusion Material Language Model
# This file is a rewrite of the complete original DPLM (DiffusionProteinLanguageModel)
# architecture: the intermediate backbone and the generation/decoding logic are kept,
# only the tokenizer is replaced by CrystaTokenizerWrapper, and hooks for embedding
# alignment are provided for training from scratch (no dplm weights are loaded).
#
# Note:
# - If the net returned by get_net(...) has a built-in embedding, it must be re-initialised at init
#   time according to the crysta vocab_size.
# - This file stays as close as possible to the original dplm.py (function names / interfaces /
#   behaviour) so that it plugs into the training pipeline seamlessly.

import math
import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer  # kept in case net_class needs them

# Project-internal registry and utility functions (kept identical to dplm.py)
from byprot.models import register_model
from byprot.models.utils import (
    LoRAConfig,
    NetConfig,
    get_net,
    get_net_class,
    sample_from_categorical,
    stochastic_sample_from_categorical,
    top_k_top_p_filtering,
    topk_masking,
)
from byprot import utils

log = utils.get_logger(__name__)


def _default_model_config_path():
    """Reference config used by :meth:`from_pretrained` to rebuild the net.

    Resolved against ``PROJECT_ROOT`` when set, otherwise against the
    repository root inferred from this file's location, so that the call does
    not depend on the current working directory.
    """
    from pathlib import Path

    root = os.environ.get("PROJECT_ROOT")
    if not root:
        # <repo>/src/byprot/models/dmlm/dmlm.py -> <repo>
        root = Path(__file__).resolve().parents[4]
    return Path(root) / "configs" / "config_all_150m_mp20.yaml"

# The CIF tokenizer / vocabulary wrapper lives in this package.
from byprot.tokenizers.crysta_tokenizer import CrystaTokenizerWrapper


# Default config dataclass (fields kept identical to DPLMConfig so configs can be swapped)
@dataclass
class DMLMConfig:
    # number of diffusion timesteps (same as DPLM)
    num_diffusion_timesteps: int = field(default=500)
    # LoRA config (used when LoRA is enabled)
    lora: LoRAConfig = field(default=LoRAConfig())
    # network config (consumed by get_net)
    net: NetConfig = field(default=NetConfig())
    # whether to enable gradient checkpointing
    gradient_ckpt: bool = field(default=False)
    # whether to enable rdm_couple coupled training
    rdm_couple: bool = field(default=False)
    # CSP debug: print corruption/condition debug info once on the first training batch
    # (this field MUST be present in the dataclass, otherwise OmegaConf.merge(default, cfg)
    #  fails on the unknown key and falls back to the raw cfg, losing defaults such as net.pretrain)
    debug_first_batch: bool = field(default=False)


# Register the model as "dmlm" so Hydra/registry can instantiate it under that name
@register_model("dmlm")
class DiffusionMaterialLanguageModel(nn.Module):
    """DMLM: a crystal language model based on the DPLM architecture.

    Notes:
    - the backbone (self.net) is created by get_net(self.cfg) (same as DPLM).
    - the tokenizer is CrystaTokenizerWrapper (the CIF tokenizer in this package).
    - when training from scratch, make sure the embedding size matches tokenizer.vocab_size:
        - if the net already contains an embedding (e.g. net.embed_tokens or net.lm_head),
          the code below marks where `maybe_resize_token_embeddings` is called to
          re-initialise that embedding.
    """

    _default_cfg = DMLMConfig()  # default config

    def __init__(self, cfg, net=None, from_dplm_weights=False, crysta_meta=None, num_diffusion_timesteps=None, **kwargs):
        """
        Args:
            cfg: OmegaConf / dict config, merged with _default_cfg
            net: use this externally provided net instance if given, otherwise it is created by get_net(self.cfg)
            from_dplm_weights: if True, try to load dplm weights (default False, because training is from scratch)
            crysta_meta: optional dict from meta.pkl, used to initialise the tokenizer when required
        """
        log.info(f'Function DiffusionMaterialLanguageModel.__init__() start.')
        self.num_diffusion_timesteps = num_diffusion_timesteps
        super().__init__()

        # merge and store the config (user cfg merged with the default cfg)
        self._update_cfg(cfg)
        log.info(f'Function DiffusionMaterialLanguageModel.__init__() Step 1: creating net.')
        # -------- 1) initialise backbone/net ----------
        # use the externally provided net if any, otherwise create it from cfg (same as DPLM)
        self.net = get_net(self.cfg) if net is None else net

        log.info(f'Function DiffusionMaterialLanguageModel.__init__() Step 2: creating tokenizer.')
        # -------- 2) initialise and replace the tokenizer ----------

        # create the crysta tokenizer instance; prefer meta (meta.pkl) when available
        try:
            # try initialising from meta (when crysta_meta was passed in)
            if crysta_meta is not None:
                self.tokenizer = CrystaTokenizerWrapper(meta=crysta_meta)
            else:
                self.tokenizer = CrystaTokenizerWrapper()
        except TypeError:
            # the wrapper does not accept a meta argument: fall back to the no-arg constructor
            self.tokenizer = CrystaTokenizerWrapper()
        log.info(f'Function DiffusionMaterialLanguageModel.__init__(): Instancing CrystaTokenizerWrapper success.')

        # bind the tokenizer to the net (overwrites the net's tokenizer to keep them consistent)
        # so that net code relying on self.net.tokenizer when producing logits uses CrystaTokenizerWrapper
        try:
            self.net.tokenizer = self.tokenizer
        except Exception:
            # ignore this if the net has no tokenizer attribute
            pass

        log.info(f'Function DiffusionMaterialLanguageModel.__init__() Step 3: special token id.')
        # -------- 3) special token ids (same as DPLM) ----------
        # these ids are expected to be provided by the net or the tokenizer
        # prefer the net (preserves the original behaviour), fall back to the tokenizer when the net has none
        self.mask_id = getattr(self.net, "mask_id", None) or getattr(self.tokenizer, "mask_token_id", None)
        self.pad_id = getattr(self.net, "pad_id", None) or getattr(self.tokenizer, "pad_token_id", None)
        self.bos_id = getattr(self.net, "bos_id", None) or getattr(self.tokenizer, "bos_token_id", None)
        self.eos_id = getattr(self.net, "eos_id", None) or getattr(self.tokenizer, "eos_token_id", None)
        #  self.x_id = getattr(self.net, "x_id", None)  # some implementations use x_id as a special placeholder

        # -------- 4) optionally load from dplm weights (not needed when training from scratch) ----------
        if from_dplm_weights:
            # the interface is kept here but unused by default. If enabled, code such as
            # get_net_class attempts to load the corresponding weights (see from_pretrained
            # in the original dplm.py for the actual loading logic, not duplicated here).
            pass

        log.info(f'Function DiffusionMaterialLanguageModel.__init__() Step 5: vocab size.')
        # -------- 5) if the net embedding does not match the crysta vocab -> re-initialise the embedding ----------
        # many net implementations contain an embedding layer, e.g. under an attribute named
        # "embed_tokens" or "embeddings.weight"; try to discover it and resize the embedding to match
        # tokenizer.vocab_size
        crysta_vocab_size = getattr(self.tokenizer, "vocab_size", None)
        if crysta_vocab_size is not None:
            # try a few common embedding attribute names
            # 1) Common HF style: net.get_input_embeddings() / net.resize_token_embeddings
            if hasattr(self.net, "resize_token_embeddings"):
                # if the net supports resize (e.g. transformers-based), call it
                log.info(f'net supports resize.')
                try:
                    self.net.resize_token_embeddings(crysta_vocab_size)
                    log.info(f'Resize token embedding Done.')
                except Exception:
                    # on failure, do not raise a fatal error, just print a hint
                    log.error(f'resize_token_embeddings failed. Please check the net embedding manually and resize it to crysta_vocab_size.')
            else:
                # 2) look for embed_tokens or embeddings directly
                log.info(f'net does not support resize.')
                if hasattr(self.net, "embed_tokens"):
                    old = self.net.embed_tokens
                    if getattr(old, "num_embeddings", None) != crysta_vocab_size:
                        # replace it with a fresh nn.Embedding and initialise it
                        hidden = old.embedding_dim if hasattr(old, "embedding_dim") else old.weight.size(1)
                        new_emb = nn.Embedding(crysta_vocab_size, hidden)
                        # use the same initialisation scheme as the original
                        nn.init.normal_(new_emb.weight, mean=0.0, std=0.02)
                        self.net.embed_tokens = new_emb
                        log.info(f"net.embed_tokens was reset to size {crysta_vocab_size} x {hidden}")
                elif hasattr(self.net, "embeddings") and hasattr(self.net.embeddings, "word_embeddings"):
                    # ESM style, or other implementations that may use embeddings.word_embeddings
                    we = self.net.embeddings.word_embeddings
                    if getattr(we, "num_embeddings", None) != crysta_vocab_size:
                        hidden = we.embedding_dim if hasattr(we, "embedding_dim") else we.weight.size(1)
                        new_we = nn.Embedding(crysta_vocab_size, hidden)
                        nn.init.normal_(new_we.weight, mean=0.0, std=0.02)
                        self.net.embeddings.word_embeddings = new_we
                        log.info(f"net.embeddings.word_embeddings was reset to size {crysta_vocab_size} x {hidden}")
                else:
                    # embedding structure not recognised: warn the user, who can adjust the net
                    # definition manually so that it matches vocab_size
                    log.warning("Warning: none of the common net embedding attributes were detected "
                                "(resize_token_embeddings/embed_tokens/embeddings.word_embeddings); "
                                "please make sure the embedding size matches tokenizer.vocab_size.")

        # -------- 6) if gradient checkpoint is enabled in the config, turn on net checkpointing (same as DPLM) ----------
        if self.cfg.gradient_ckpt:
            if hasattr(self.net, "supports_gradient_checkpointing"):
                self.net.supports_gradient_checkpointing = True
                try:
                    # some model APIs support gradient_checkpointing_enable()
                    self.net.gradient_checkpointing_enable()
                except Exception:
                    pass
        log.info(f'Function DiffusionMaterialLanguageModel.__init__() Done.')


    # from_pretrained interface kept consistent with DPLM (retained for future pretrained-weight loading)
    @classmethod
    def from_pretrained(
        cls, net_name, cfg_override={}, net_override={}, from_huggingface=False
    ):
        """
        Mirrors DPLM's from_pretrained implementation: the interface is kept so that a
        checkpoint can be loaded in the future.
        """
        from pathlib import Path
        from collections import OrderedDict
        import json
        import torch

        if not from_huggingface:
            # Build an empty model with the architecture described by the
            # reference config, then load the weights into it. The config path
            # is resolved against the repository root (PROJECT_ROOT or the
            # location of this file) so the call works from any working dir.
            from byprot.utils.config import load_yaml_config

            cfg_path = _default_model_config_path()
            full_cfg = load_yaml_config(str(cfg_path))
            cfg = load_yaml_config(str(cfg_path)).model
            cfg.net.pretrain = False
            if "_target_" in cfg:
                cfg.pop("_target_")
            model = cls(cfg)

            pretrained_state_dict = torch.load(
                net_name, map_location=torch.device("cpu")
            )["state_dict"]
            new_pretrained_state_dict = OrderedDict()

            # remove the "model." prefix if present
            for k, v in pretrained_state_dict.items():
                new_pretrained_state_dict[k[6:]] = v

            missing, unexpected = model.load_state_dict(
                new_pretrained_state_dict, strict=False
            )
            print(
                f"Restored from {net_name} with {len(missing)} missing and {len(unexpected)} unexpected keys"
            )
            if len(missing) > 0:
                print(f"Missing Keys: {missing}")
                print(f"Unexpected Keys: {unexpected}")
            return model
        else:
            # if the network must be loaded from HuggingFace or a local HF mirror (interface kept)
            # this example uses the local_dir route (as in dplm.py); adapt it as needed
            local_dir = "airkingbd/dmlm_650m"  # point this at your own path if a local HF-style repo exists
            if local_dir is None:
                raise ValueError(
                    "`local_dir` must be provided when `from_huggingface=True` and server cannot access HuggingFace."
                )

            config_path = Path(local_dir, "config.json")
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found at {config_path}")

            with open(config_path, "r") as f:
                config = json.load(f)
            dplm_type = config.get("dplm_type")  # keep the field name for compatibility with the original implementation (may need renaming)
            if dplm_type is None:
                raise ValueError("`dplm_type` not found in config.json")

            net_class = get_net_class(dplm_type)
            net = net_class.from_pretrained(str(local_dir), **net_override)

            return cls(cfg=cfg_override, net=net)

    # merge the config (identical to DPLM's _update_cfg)
    def _update_cfg(self, cfg):
        # # original code:
        # self.cfg = OmegaConf.merge(self._default_cfg, cfg)

        # changed to:
        try:
            self.cfg = OmegaConf.merge(self._default_cfg, cfg)
        except Exception as e:
            print(f"Config merge failed: {e}")
            print("Using the file config, ignoring the default config")
            self.cfg = cfg  # use the file config directly

    # The functions below (q_sample_coupled / q_sample / forward / compute_loss / generate, ...)
    # largely keep DPLM's original implementation, with line-by-line comments for readability.
    # ---- q_sample_coupled ----
    def q_sample_coupled(self, x_0, t1, t2, maskable_mask, condition_mask=None):
        # t1_eq_t2_mask marks the sequences whose two timesteps are equal (used by the coupling strategy)
        t1_eq_t2_mask = t1 == t2
        # normalise t1, t2 so that t1 >= t2
        t1, t2 = torch.maximum(t1, t2).float(), torch.minimum(t1, t2).float()

        # sample t1
        u = torch.rand_like(x_0, dtype=torch.float)
        # for each position, mask with probability (t1/num_timesteps) (i.e. replace by mask_id)
        t1_mask = (
            u < (t1 / self.cfg.num_diffusion_timesteps)[:, None]
        ) & maskable_mask
        if condition_mask is not None:
            t1_mask = t1_mask & ~condition_mask
        # replace the selected positions with mask_id to obtain x_t1
        x_t1 = x_0.masked_fill(t1_mask, self.mask_id)
        if condition_mask is not None:
            # CSP: composition tokens always keep x_0 (double safeguard, even if maskable_mask missed them)
            x_t1 = torch.where(condition_mask, x_0, x_t1)

        # sample t2
        u = torch.rand_like(x_0, dtype=torch.float)
        # among the positions already marked by t1_mask, keep a proportional subset in t2
        t2_mask = t1_mask & (u > ((t1 - t2) / t1)[:, None])
        u = torch.rand_like(x_0[t1_eq_t2_mask], dtype=torch.float)
        # handle the t1 == t2 case with a special rule
        t2_mask[t1_eq_t2_mask] = (
            u < (t1[t1_eq_t2_mask] / self.cfg.num_diffusion_timesteps)[:, None]
        ) & (maskable_mask[t1_eq_t2_mask])
        if condition_mask is not None:
            t2_mask[t1_eq_t2_mask] = t2_mask[t1_eq_t2_mask] & ~condition_mask[t1_eq_t2_mask]
        x_t2 = x_0.masked_fill(t2_mask, self.mask_id)
        if condition_mask is not None:
            x_t2 = torch.where(condition_mask, x_0, x_t2)

        # return the concatenated result: x_t (both batch halves concatenated on dim 0), t (timesteps)
        # and the mask
        return {
            "x_t": torch.cat([x_t1, x_t2], dim=0),
            "t": torch.cat([t1, t2]),
            "mask_mask": torch.cat([t1_mask, t2_mask], dim=0),
        }

    # ---- q_sample ----
    def q_sample(self, x_0, t1, maskable_mask, condition_mask=None):
        # sample t1
        u = torch.rand_like(x_0, dtype=torch.float)
        t1_mask = (
            u < (t1 / self.cfg.num_diffusion_timesteps)[:, None]
        ) & maskable_mask
        if condition_mask is not None:
            t1_mask = t1_mask & ~condition_mask
        x_t1 = x_0.masked_fill(t1_mask, self.mask_id)
        # note: the original dplm performs masked_fill twice here (possibly a typo or
        # redundant); kept for compatibility
        x_t1 = x_t1.masked_fill(t1_mask, self.mask_id)
        if condition_mask is not None:
            # CSP: composition tokens keep x_0 at any timestep (double safeguard)
            x_t1 = torch.where(condition_mask, x_0, x_t1)

        return {
            "x_t": x_t1,
            "t": t1,
            "mask_mask": t1_mask,
        }

    # ---- forward: use the net to produce logits ----
    def forward(self, input_ids, return_last_hidden_state=False, **kwargs):
        # the net interface matches DPLM: pass input_ids and get a dict containing "logits"
        # and optionally "last_hidden_state"
        outputs = self.net(
            input_ids=input_ids,
        )
        logits = outputs["logits"]
        if return_last_hidden_state:
            last_hidden_state = outputs["last_hidden_state"]
            return logits, last_hidden_state
        else:
            return logits

    # ---- compute_loss: sampling + loss computation used during training ----
    def compute_loss(self, batch, weighting="constant", condition_mask=None):
        # batch is expected to contain "targets" (the ground-truth token ids)
        # condition_mask: [B, L] bool, where True marks composition condition positions (CSP):
        #   those positions are not corrupted by diffusion and do not contribute to the loss.
        #   Random generation tasks do not pass it (None), so the behaviour is exactly as before.
        target = batch["targets"]
        batch_size = target.size(0)
        # randomly sample two timesteps t1, t2 (length 2*B, then chunked into two vectors)
        t1, t2 = torch.randint(
            1,
            self.cfg.num_diffusion_timesteps + 1,
            (2 * target.size(0),),
            device=target.device,
        ).chunk(2)
        # CSP: composition positions are not diffusion variables (never masked, never in the loss)
        if condition_mask is not None:
            if condition_mask.dtype != torch.bool:
                condition_mask = condition_mask.bool()
            if condition_mask.shape != target.shape:
                raise ValueError(
                    f"condition_mask shape {tuple(condition_mask.shape)} != "
                    f"target shape {tuple(target.shape)}"
                )
            condition_mask = condition_mask.to(target.device)
        maskable_mask = self.get_non_special_symbol_mask(target)
        if condition_mask is not None:
            maskable_mask = maskable_mask & ~condition_mask

        # if rdm_couple is enabled, use the coupled-sample strategy (matching the paper/implementation)
        if self.cfg.rdm_couple:
            print("  🔄 Using the q_sample_coupled strategy")
            x_t, t, loss_mask = list(
                self.q_sample_coupled(
                    target,
                    t1,
                    t2,
                    maskable_mask=maskable_mask,
                    condition_mask=condition_mask,
                ).values()
            )
            print(f"    x_t shape: {x_t.shape}")
            print(f"    t shape: {t.shape}")
            print(f"    loss_mask shape: {loss_mask.shape}")
            # the target must be repeated as well to match x_t's batch dimension
            # (coupling doubles the batch)
            target = target.repeat(2, 1)
        else:
            # otherwise use the plain q_sample
            # print("  🔄 Using the plain q_sample strategy")
            x_t, t, loss_mask = list(
                self.q_sample(
                    target,
                    t1,
                    maskable_mask=maskable_mask,
                    condition_mask=condition_mask,
                ).values()
            )
        # forward pass to obtain the logits
        logits = self.forward(x_t)

        # CSP debug output: printed only once, on the first batch and only on rank 0,
        # to confirm that the composition span is detected correctly; nothing afterwards.
        if (
            condition_mask is not None
            and getattr(self.cfg, "debug_first_batch", False)
            and not getattr(self, "_csp_first_batch_printed", False)
        ):
            try:
                import torch.distributed as dist

                is_rank0 = (
                    not dist.is_initialized() or dist.get_rank() == 0
                )
            except Exception:
                is_rank0 = True
            if is_rank0:
                self._print_csp_first_batch(
                    target, x_t, loss_mask, condition_mask
                )
            self._csp_first_batch_printed = True

        # compute the per-timestep weight (linear or constant)
        num_timesteps = self.cfg.num_diffusion_timesteps
        weight = {
            "linear": (
                num_timesteps - (t - 1)
            ),  # num_timesteps * (1 - (t-1)/num_timesteps)
            "constant": num_timesteps * torch.ones_like(t),
        }[weighting][:, None].float() / num_timesteps

        # explicitly exclude condition positions from the masked loss (double safeguard;
        # because condition positions are never masked this is usually a no-op, but it
        # prevents label smoothing / auxiliary loss / weighting from pulling condition
        # tokens in)
        if condition_mask is not None:
            if self.cfg.rdm_couple:
                loss_mask = loss_mask & ~condition_mask.repeat(2, 1)
            else:
                loss_mask = loss_mask & ~condition_mask

        # return logits, target, loss_mask and the weight (the training loop uses these to compute the loss)
        return logits, target, loss_mask, weight

    # ---- _print_csp_first_batch: key CSP debug output for the first batch ----
    def _print_csp_first_batch(self, target, x_t, loss_mask, condition_mask):
        """Print a one-off confirmation of the composition span on the first batch of CSP training."""
        tok = self.tokenizer
        try:
            b0 = target[0].detach().cpu()
            cond0 = condition_mask[0].detach().cpu()
            x0_t = x_t[0].detach().cpu()
            lm0 = loss_mask[0].detach().cpu()
            nl = int(torch.nonzero(cond0).max()) if cond0.any() else -1

            def trunc(x):
                return x[:80].tolist()

            print("\n" + "=" * 70)
            print("[CSP first-batch debug] composition span check")
            print("Raw CIF head:      ", repr(tok.decode(trunc(b0))))
            print("Token ids head:    ", trunc(b0))
            print("Condition tokens:  ", repr(tok.decode(b0[cond0].tolist())))
            print("Condition ids:     ", b0[cond0].tolist())
            print("Condition count:   ", int(cond0.sum()), " (mask ends at idx", nl, ")")
            print("Condition mask[0:30]:", cond0[:30].tolist())
            print("Corrupted ids head:", trunc(x0_t))
            print("Corrupted decode:  ", repr(tok.decode(trunc(x0_t))))
            # key assertion: after corruption at any timestep the composition tokens are completely unchanged
            assert torch.equal(
                x0_t[cond0], b0[cond0]
            ), "composition tokens changed after corruption!"
            assert not (lm0 & cond0).any(), "condition positions leaked into loss_mask!"
            print("Loss positions[0:30]:", lm0[:30].tolist())
            print(f"Loss positions count: {int(lm0.sum())} / {lm0.numel()}")
            print("Assertions passed: composition unchanged & excluded from loss.")
            print("=" * 70 + "\n")
        except Exception as e:
            print(f"[CSP first-batch debug] skipped due to error: {e}")

    # ---- forward_encoder: left empty, extend as needed ----
    def forward_encoder(self, input_tokens, **kwargs):
        # override this method in a subclass if encoder-conditional generation is needed
        return {}

    # ---- initialize_output_tokens: build the initial output tokens (fill the positions to predict with mask_id) ----
    def initialize_output_tokens(self, input_tokens, partial_masks=None, **kwargs):
        tokens = input_tokens
        if tokens is None:
            raise NotImplementedError
        else:
            # mask of the positions that may be predicted (non-special symbols)
            output_mask = self.get_non_special_symbol_mask(tokens, partial_masks=partial_masks)

            # replace those positions with mask_id as the initialisation
            output_tokens = tokens.masked_fill(output_mask, self.mask_id)
            # initial scores are all 0
            output_scores = torch.zeros_like(output_tokens, dtype=torch.float)

            return output_tokens, output_scores

    # ---- resample: rejection sampling to remove repetitive token patterns ----
    def resample(self, _tokens, _scores, ratio, scale):
        """Rejection sampling to reduce repetitive tokens (e.g., 'VVVVV...')"""

        to_be_resample_idx = []
        resample_input = []
        resample_input_mask = []
        resample_input_scores = []

        # record the positions of each token in every sequence and find the most frequent token
        for i, seq in enumerate(_tokens):
            most_token_dict = {}
            most_token_num = -1
            for j, token in enumerate(seq):
                token = int(token)
                if token not in most_token_dict:
                    most_token_dict[token] = [j]
                else:
                    most_token_dict[token].append(j)
                if len(most_token_dict[token]) > most_token_num:
                    most_token_num = len(most_token_dict[token])
            # if a token occurs more often than the threshold (len(seq) * ratio), mark those positions
            # as needing resampling
            if most_token_num > len(seq) * ratio:
                to_be_resample_idx.append(i)
                resample_input_scores.append(_scores[i])
                mask = torch.zeros_like(seq).bool()
                for k, v in most_token_dict.items():
                    if len(v) > len(seq) * ratio:
                        mask |= seq.eq(k)
                resample_input_mask.append(mask)
                resample_input.append(seq.masked_fill(mask, self.mask_id))

        # if there are sequences that need resampling
        if len(to_be_resample_idx) > 0:
            # stack the sequences to resample into a batch and cast them back to the same dtype
            resample_input = torch.stack(resample_input, dim=0).type_as(
                _tokens
            )
            resample_input_scores = torch.stack(
                resample_input_scores, dim=0
            ).type_as(_scores)
            resample_input_mask = (
                torch.stack(resample_input_mask, dim=0).type_as(_tokens).bool()
            )
            # re-predict the logits with the net
            resample_logits = self.net(
                input_ids=resample_input,
            )["logits"]
            # keep the dtypes consistent
            if resample_logits.dtype != _scores.dtype:
                resample_logits = resample_logits.type_as(_scores)
            # set the logits of special tokens to -inf so they are never sampled
            resample_logits[..., self.mask_id] = -math.inf
#            resample_logits[..., self.x_id] = -math.inf
            resample_logits[..., self.pad_id] = -math.inf
            resample_logits[..., self.bos_id] = -math.inf
            resample_logits[..., self.eos_id] = -math.inf

            # apply top-k/top-p filtering
            resample_logits = top_k_top_p_filtering(
                resample_logits, top_p=0.95
            )
            noise_scale = scale
            assert resample_logits.size(0) == len(to_be_resample_idx)
            (
                resample_tokens,
                resample_scores,
            ) = stochastic_sample_from_categorical(
                resample_logits, temperature=0.0, noise_scale=noise_scale
            )
            # write the resampled results back to the original positions
            resample_input.masked_scatter_(
                resample_input_mask, resample_tokens[resample_input_mask]
            )
            resample_input_scores.masked_scatter_(
                resample_input_mask, resample_scores[resample_input_mask]
            )
            _tokens[to_be_resample_idx], _scores[to_be_resample_idx] = (
                resample_input,
                resample_input_scores,
            )

    # ---- forward_decoder: a decoder step (used for generation) ----
    def forward_decoder(
        self,
        prev_decoder_out,
        encoder_out=None,
        need_attn_weights=False,
        partial_masks=None,
        sampling_strategy="gumbel_argmax",
        disable_resample=True,
        resample_ratio=0.25,
    ):
        # copy the input state so that in-place edits do not affect the caller
        output_tokens = prev_decoder_out["output_tokens"].clone()
        output_scores = prev_decoder_out["output_scores"].clone()
        step, max_step = prev_decoder_out["step"], prev_decoder_out["max_step"]
        temperature = prev_decoder_out["temperature"]
        history = prev_decoder_out["history"]

        # compute the positions that can currently be predicted (non-special symbols)
        output_masks = self.get_non_special_symbol_mask(
            output_tokens, partial_masks=partial_masks
        )

        # call the net to get the logits (the model's main interface)
        net_out = self.net(
            input_ids=output_tokens,
        )

        logits = net_out["logits"]
        attentions = net_out["attentions"] if need_attn_weights else None

        # dtype alignment: make logits match output_scores for later comparison/sorting
        if logits.dtype != output_scores.dtype:
            logits = logits.type_as(output_scores)

        # set the logits of special tokens to -inf so the model never generates them
        logits[..., self.mask_id] = -math.inf
#        logits[..., self.x_id] = -math.inf
        logits[..., self.pad_id] = -math.inf
        logits[..., self.bos_id] = -math.inf
        logits[..., self.eos_id] = -math.inf

        # pick tokens according to the sampling strategy
        if sampling_strategy == "vanilla":
            _tokens, _scores = sample_from_categorical(
                logits, temperature=temperature
            )
        elif sampling_strategy == "argmax":
            # take the maximum directly
            _scores, _tokens = logits.max(-1)
        elif sampling_strategy == "gumbel_argmax":
            # approximate randomised sampling with Gumbel + argmax
            noise_scale = 1.0
            _tokens, _scores = stochastic_sample_from_categorical(
                logits, temperature=0.0, noise_scale=noise_scale
            )

            if not disable_resample:
                # if resampling is allowed, call rejection sampling to remove repetitive patterns
                self.resample(
                    _tokens, _scores, ratio=resample_ratio, scale=1.0
                )
        else:
            raise NotImplementedError

        # write back only at the predicted positions (masked_scatter_ only replaces output_masks)
        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])

        # save the history
        history.append(output_tokens.clone())

        return dict(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attentions=attentions,  # may contain attention weights
            step=step + 1,
            max_step=max_step,
            history=history,
            hidden_states=net_out.get("last_hidden_state", None),
        )

    # ---- get_non_special_symbol_mask: compute the non-special-token mask ----
    def get_non_special_symbol_mask(self, output_tokens, partial_masks=None):
        non_special_sym_mask = (
            output_tokens.ne(self.pad_id)
            & output_tokens.ne(self.bos_id)
            & output_tokens.ne(self.eos_id)
        )
        if partial_masks is not None:
            non_special_sym_mask &= ~partial_masks
        return non_special_sym_mask

    # ---- _reparam_decoding: the reparam decoding strategy (complex top-k / stochastic implementation) ----
    def _reparam_decoding(
        self,
        output_tokens,
        output_scores,
        cur_tokens,
        cur_scores,
        decoding_strategy,
        xt_neq_x0,
        non_special_sym_mask,
        t,
        max_step,
        noise,
    ):
        """This function is used to perform reparameterized decoding."""
        # decoding_strategy format: "reparam-<conditioning>-<topk_mode>-<schedule>"
        _, condition, topk_mode, schedule = decoding_strategy.split("-")

        # compute the denoising rate from the schedule
        if schedule == "linear":
            rate = 1 - t / max_step
        elif schedule == "cosine":
            rate = np.cos(t / max_step * np.pi * 0.5)
        else:
            raise NotImplementedError

        # cutoff length for top-k = number of non-special tokens * rate
        cutoff_len = (
            non_special_sym_mask.sum(1, keepdim=True).type_as(output_scores)
            * rate
        ).long()
        # give special tokens a large score so that they are never selected
        _scores_for_topk = cur_scores.masked_fill(
            ~non_special_sym_mask, 1000.0
        )

        # two top-k modes: stochastic (with Gumbel noise) or deterministic
        if topk_mode.startswith("stochastic"):
            noise_scale = float(topk_mode.replace("stochastic", ""))
            lowest_k_mask = topk_masking(
                _scores_for_topk,
                cutoff_len,
                stochastic=True,
                temp=noise_scale * rate,
            )
        elif topk_mode == "deterministic":
            lowest_k_mask = topk_masking(
                _scores_for_topk, cutoff_len, stochastic=False
            )
        else:
            raise NotImplementedError

        # compute not_v1_t from condition (cond/uncond); it is related to the top-k strategy
        if condition == "cond":
            not_v1_t = (
                (cur_tokens == output_tokens)
                & (cur_scores < output_scores)
                & lowest_k_mask
            )
        elif condition == "uncond":
            not_v1_t = lowest_k_mask
        else:
            raise NotImplementedError

        # handle the positions with b_t = 0 (set them to noise if they are inside lowest_k)
        not_v2_t = lowest_k_mask

        last_mask_position = xt_neq_x0
        masked_to_noise = (~xt_neq_x0 & not_v1_t) | (xt_neq_x0 & not_v2_t)
        # assign noise (a tensor or a scalar) to the masked_to_noise positions
        if isinstance(noise, torch.Tensor):
            output_tokens.masked_scatter_(
                masked_to_noise, noise[masked_to_noise]
            )
        elif isinstance(noise, (int, float)):
            output_tokens.masked_fill_(masked_to_noise, noise)
        else:
            raise NotImplementedError(
                "noise should be either a tensor or a scalar"
            )
        # set the scores at the corresponding positions to -inf
        output_scores.masked_fill_(masked_to_noise, -math.inf)

        # masked_to_x0 marks the positions set to the current cur_tokens
        masked_to_x0 = xt_neq_x0 & ~not_v2_t
        output_tokens.masked_scatter_(masked_to_x0, cur_tokens[masked_to_x0])
        output_scores.masked_scatter_(masked_to_x0, cur_scores[masked_to_x0])
        assert ((masked_to_x0 & last_mask_position) == masked_to_x0).all()

        # compute and return the next not_b_t (saved for the next step)
        new_xt_neq_x0 = (xt_neq_x0 | not_v1_t) & not_v2_t
        assert (new_xt_neq_x0 == not_v2_t).all()
        return new_xt_neq_x0, output_tokens, output_scores

    # ---- generate: high-level generation loop (init, decoder steps, reparam strategy) ----
    def generate(
        self,
        input_tokens,
        tokenizer=None,
        max_iter=None,
        temperature=None,
        partial_masks=None,
        sampling_strategy="gumbel_argmax",
        disable_resample=False,
        resample_ratio=0.25,
        condition_mask=None,
        condition_ids=None,
    ):
        # keep the interface style: tokenizer / max_iter / temperature are accepted,
        # default behaviour matches DPLM
        #
        # condition_mask: [B, L] bool (CSP). True marks composition condition tokens:
        #   no remasking / sampling logic may modify them.
        # condition_ids:  [B, L] long, the original ids of the condition tokens (usually input_tokens).
        # Random generation tasks pass neither of them -> the behaviour is exactly as before.
        tokenizer = tokenizer
        max_iter = max_iter
        temperature = temperature

        if condition_mask is not None:
            condition_mask = condition_mask.to(input_tokens.device)
            if condition_ids is None:
                condition_ids = input_tokens
            # condition positions must be protected by partial_masks (never initially
            # masked / never resampled)
            if partial_masks is None:
                partial_masks = condition_mask
            else:
                partial_masks = partial_masks.to(input_tokens.device) | condition_mask

        # 0) encoder (optional)
        encoder_out = self.forward_encoder(input_tokens)
        # 1) initialise output tokens (fill the positions to predict with mask)
        (
            initial_output_tokens,
            initial_output_scores,
        ) = self.initialize_output_tokens(
            input_tokens, encoder_out=encoder_out, partial_masks=partial_masks
        )
        prev_decoder_out = dict(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            output_masks=None,
            attentions=None,
            step=0,
            max_step=max_iter,
            history=[initial_output_tokens.clone()],
            temperature=temperature,
        )

        # compute the initial output_masks
        prev_decoder_out["output_masks"] = self.get_non_special_symbol_mask(
            prev_decoder_out["output_tokens"], partial_masks=partial_masks
        )

        # iterate the decoding steps
        # for step in tqdm(range(max_iter), desc="Decoding"):
        for step in range(max_iter):

            # 2.1: predict
            with torch.no_grad():
                decoder_out = self.forward_decoder(
                    prev_decoder_out=prev_decoder_out,
                    encoder_out=encoder_out,
                    partial_masks=partial_masks,
                    sampling_strategy=sampling_strategy,
                    disable_resample=disable_resample,
                    resample_ratio=resample_ratio,
                )

            output_tokens = decoder_out["output_tokens"]
            output_scores = decoder_out["output_scores"]

            # 2.2: re-mask the low-confidence part and apply the reparam decoding strategy
            non_special_sym_mask = self.get_non_special_symbol_mask(
                prev_decoder_out["output_tokens"], partial_masks=partial_masks
            )

            (
                output_masks,
                result_tokens,
                result_scores,
            ) = self._reparam_decoding(
                output_tokens=prev_decoder_out["output_tokens"].clone(),
                output_scores=prev_decoder_out["output_scores"].clone(),
                cur_tokens=output_tokens.clone(),
                cur_scores=output_scores.clone(),
                decoding_strategy="reparam-uncond-deterministic-linear",
                xt_neq_x0=prev_decoder_out["output_masks"],
                non_special_sym_mask=non_special_sym_mask,
                t=step + 1,
                max_step=max_iter,
                noise=self.mask_id,
            )

            prev_decoder_out.update(output_masks=output_masks)
            output_tokens = result_tokens
            output_scores = result_scores

            if condition_mask is not None:
                # CSP double safeguard: after every denoising step, force the composition positions
                # back to the condition tokens so that no remasking / sampling logic can change
                # the condition by accident.
                output_tokens = output_tokens.masked_scatter(
                    condition_mask, condition_ids[condition_mask]
                )

            prev_decoder_out.update(
                output_tokens=output_tokens,
                output_scores=output_scores,
                step=step + 1,
                history=decoder_out["history"],
            )

        decoder_out = prev_decoder_out
        # return the final generated token matrix
        return decoder_out["output_tokens"]
