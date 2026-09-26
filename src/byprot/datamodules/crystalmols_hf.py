# Copyright (c) 2025
# SPDX-License-Identifier: Apache-2.0

import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

from byprot.tokenizers.crysta_tokenizer import CrystaTokenizerWrapper
from byprot.datamodules import register_datamodule
from byprot import utils

log = utils.get_logger(__name__)


def default_data_root():
    """Return the bundled ``data-bin/crystalmols`` directory.

    Resolved against ``PROJECT_ROOT`` when set, otherwise against the
    repository root inferred from this file's location, so the defaults below
    do not depend on the current working directory.
    """
    root = os.environ.get("PROJECT_ROOT")
    if not root:
        # <repo>/src/byprot/datamodules/crystalmols_hf.py -> <repo>
        root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
    return os.path.join(str(root), "data-bin", "crystalmols")


def build_composition_condition_mask(seq, data_token_id, newline_token_id):
    """Build the CSP condition mask for one tokenized CIF row.

    Every training row is a tokenized CIF whose first line is the composition,
    e.g. ``data_NaCl\\nloop_...``. The CSP condition spans the leading
    ``data_ <element/number tokens> \\n`` prefix, i.e. every position up to and
    including the first ``\\n`` token. Those positions must always keep their
    original tokens during training corruption and during diffusion sampling.

    Args:
        seq: 1-D long tensor, a single unpadded token id sequence.
        data_token_id: token id of ``data_`` (taken from the vocab, never hardcoded).
        newline_token_id: token id of ``\\n``.

    Returns:
        A bool tensor of the same length as ``seq``. All-False when the row does
        not start with ``data_`` or contains no newline (such a row degenerates
        into an unconditional sample; not expected in practice).
    """
    mask = torch.zeros(len(seq), dtype=torch.bool)
    if seq.numel() == 0:
        return mask
    if data_token_id is None or newline_token_id is None:
        return mask
    if seq[0] != data_token_id:
        return mask
    nl_pos = (seq == newline_token_id).nonzero()
    if nl_pos.numel() == 0:
        return mask
    first_nl = int(nl_pos[0])
    # The condition includes the trailing '\n', matching the data_<formula>\n
    # prefix used at the generation-prompt stage.
    mask[: first_nl + 1] = True
    return mask


class CrystalDataset(Dataset):
    """Dataset for tokenized CIFs (train.bin / val.bin / meta.pkl)."""

    @staticmethod
    def _resolve_start_pkl(data_file):
        """Locate the ``starts`` pkl that belongs to ``data_file``.

        ``<dir>/tokens_<name>/<train|val>.bin`` maps to
        ``<dir>/tokens_<name>/starts_<name>_<train|val>.pkl``. If that exact
        name is absent, fall back to the single ``starts_*_<split>.pkl`` in the
        same directory. Returns ``None`` when there is none, in which case
        ``__init__`` recovers row boundaries from the ``data_`` token in the bin.
        """
        d = os.path.dirname(data_file)
        split = os.path.basename(data_file).split(".")[0]
        base = os.path.basename(d)
        if base.startswith("tokens_"):
            cand = os.path.join(d, f"starts_{base[len('tokens_'):]}_{split}.pkl")
            if os.path.exists(cand):
                return cand
        import glob
        cands = sorted(glob.glob(os.path.join(d, f"starts_*_{split}.pkl")))
        if len(cands) == 1:
            return cands[0]
        if len(cands) > 1:
            raise FileNotFoundError(f"ambiguous starts pkls {cands} for {data_file}")
        return None

    def __init__(
        self,
        data_file: str = None,
        max_len: int = 2048,
        pad_token_id: int = 376,
        data_token_id: int = None,  # token id of 'data_'; used to auto-locate CIF starts when no starts pkl exists
    ):
        log.info(f'Function CrystalDataset.__init__() Start.')
        super().__init__()
        if data_file is None:
            data_file = os.path.join(default_data_root(), "tokens_mp_20")
        self.max_len = max_len
        self.data_token_id = data_token_id
        # self.tokenizer = tokenizer
        # self.pad_token_id = pad_token_id

        # Resolve the matching starts pkl next to data_file (all datasets follow
        # the same rule, so no per-dataset path table is needed).
        start_file = self._resolve_start_pkl(data_file)

        # load tokenized sequences (uint16 array)

        with open(data_file, "rb") as f:
            self.samples = np.frombuffer(f.read(), dtype=np.uint16)

        if start_file is not None:
            with open(start_file, "rb") as f:
                starts = np.array(pickle.load(f), dtype=np.int64)
            if starts.size == 0 or int(starts[0]) != 0:
                raise ValueError(f"starts pkl {start_file} does not begin at 0: {starts[:5]}")
        else:
            # No starts pkl (e.g. val.bin of carbon_24/mpts_52/perov_5):
            # same rule as tokens_mp_20/extract_starts.py -- every 'data_' token in
            # the bin marks the start of a CIF (verified to match the pkl exactly on mp_20).
            if self.data_token_id is None:
                raise ValueError(
                    f"CrystalDataset: no starts pkl found ({start_file}) and no "
                    f"data_token_id given to auto-split {data_file}"
                )
            starts = np.flatnonzero(self.samples == self.data_token_id)
            if starts.size == 0 or int(starts[0]) != 0:
                raise ValueError(
                    f"data_ token ({self.data_token_id}) auto-split failed: "
                    f"starts.size={starts.size}, starts[0]={starts[0] if starts.size else None}"
                )
            log.info(
                f"CrystalDataset: no starts pkl; auto-split {data_file} by data_ token -> {starts.size} rows"
            )

        # (start, end) row ranges: one row per CIF, the last row extends to EOF
        ends = np.empty_like(starts)
        ends[:-1] = starts[1:]
        ends[-1] = self.samples.size
        lengths = ends - starts

        # max_len filter: drop over-long samples (e.g. the >1024-token sequences in
        # mpts_52, which exceed ESM's max_position_embeddings); max_len<=0 disables it
        if max_len and max_len > 0:
            keep = lengths <= max_len
            if not bool(keep.all()):
                log.info(
                    f"CrystalDataset: dropped {(~keep).sum()} rows (len>{max_len}), "
                    f"keeping {int(keep.sum())}/{starts.size} rows"
                )
            starts = starts[keep]
            ends = ends[keep]
        self._starts = starts
        self._ends = ends
        self.num_samples = len(starts)
        log.info(f'Function CrystalDataset.__init__() Done. rows={self.num_samples}')

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

        start = int(self._starts[idx])
        end = int(self._ends[idx])

        data_slice = self.samples[start:end]
        # Make a fully independent copy: np.array() allocates a new array, and
        # torch.tensor() copies (unlike torch.from_numpy(), which may share memory).
        tokens_np = np.array(data_slice, dtype=np.int64)
        tokens = torch.tensor(tokens_np, dtype=torch.long)
        tokens = tokens.contiguous()
        return {
            "input_ids": tokens,
            "targets": tokens.clone(),
            "input_mask": torch.ones(len(tokens), dtype=torch.bool).contiguous()
        }
        # return torch.tensor(tokens, dtype=torch.long)


@register_datamodule("crystalmols_hf")
class CrystalMolsDataModule(pl.LightningDataModule):
    """Lightning DataModule for CrystaLLM tokenized CIF datasets."""

    def __init__(
            self,
            data_dir: str = None,
            max_tokens=8000,
            max_len=2048,
            num_workers=8,
            tokenizer="crystallm",
            vocab_file=None,
            batch_size=8,
            conditioning: str = "none",  # 'none' = random generation (unconditional) / 'composition' = CSP
            special_tokens=None,  # accepted for compatibility, unused
            **kwargs,  # absorb any other unexpected Hydra keys instead of failing
    ):
        log.info(f'Function CrystalMolsDataModule__init__ Start!')
        super().__init__()

        log.info(f'Function CrystalMolsDataModule__init__: Step 1: Initialize parameter!')
        if data_dir is None:
            data_dir = os.path.join(default_data_root(), "tokens_mp_20")
        if not isinstance(data_dir, (str, os.PathLike)):
            from omegaconf import OmegaConf
            try:
                data_dir = OmegaConf.to_container(data_dir, resolve=True)
                if isinstance(data_dir, dict):
                    # if data_dir is {"path": "..."}, take the path out of it
                    data_dir = list(data_dir.values())[0]
                data_dir = str(data_dir)
            except Exception:
                data_dir = str(data_dir)
        self.data_dir = data_dir
        self.max_tokens = max_tokens
        self.max_len = max_len
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.conditioning = conditioning  # 'none' | 'composition'
        self.special_tokens = special_tokens  # stored but unused (avoids conflicts)

        log.info(f'Function CrystalMolsDataModule__init__: Step 2: Load meta.pkl!')
        # Load meta.pkl
        meta_file = os.path.join(self.data_dir, "meta.pkl")
        if not os.path.exists(meta_file):
            raise FileNotFoundError(f"meta.pkl not found in {self.data_dir}")
        with open(meta_file, "rb") as f:
            self.meta = pickle.load(f)

        log.info(f'Function CrystalMolsDataModule__init__: Step 3: Initialize tokenizer!')
        # Initialize tokenizer
        if tokenizer == "crystallm":
            self.tokenizer = CrystaTokenizerWrapper(meta=self.meta)
        else:
            raise ValueError(f"Unsupported tokenizer: {tokenizer}")

        # Special token ids from meta.pkl
        self.pad_token_id = self.meta.get("pad_token_id", 371)
        self.eos_token_id = self.meta.get("eos_token_id", 372)
        self.vocab_size = self.meta.get("vocab_size", 375)

        # CSP: resolve the data_ / '\n' token ids from the vocabulary instead of
        # hardcoding them; used by collate to build the per-sample condition_mask.
        self._data_token_id = self.tokenizer._token_to_id.get("data_")
        self._newline_token_id = self.tokenizer._token_to_id.get("\n")
        if self.conditioning == "composition" and (
            self._data_token_id is None or self._newline_token_id is None
        ):
            log.warning(
                "conditioning='composition' but 'data_' or '\\n' not found in vocab; "
                "condition_mask will be all-False!"
            )
        self.train_dataset = None  # cached Dataset
        self.val_dataset = None
        log.info(f'Function CrystalMolsDataModule__init__ Done!')

    def setup(self, stage=None):
        if self.train_dataset is None:
            train_file = os.path.join(self.data_dir, "train.bin")
            self.train_dataset = CrystalDataset(
                train_file, self.max_len, self.pad_token_id,
                data_token_id=self._data_token_id,  # auto-split when no starts pkl (all four datasets)
            )
        if self.val_dataset is None:
            val_file = os.path.join(self.data_dir, "val.bin")
            self.val_dataset = CrystalDataset(
                val_file, self.max_len, self.pad_token_id,
                data_token_id=self._data_token_id,
            )

    def _create_collate_fn(self):
        """Build the collate function."""
        # CSP: composition span lengths differ across samples in a batch, so the
        # condition_mask must be a per-sample [B, L] bool mask.
        conditioning = self.conditioning
        data_token_id = self._data_token_id
        newline_token_id = self._newline_token_id

        def collate_fn(batch):
            if not batch:
                return {}

            # longest sequence length in this batch
            max_len = max(item["input_ids"].size(0) for item in batch)
            batch_size = len(batch)

            # allocate brand-new tensors
            input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
            targets = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
            attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

            for i, item in enumerate(batch):
                seq_len = item["input_ids"].size(0)
                input_ids[i, :seq_len] = item["input_ids"]
                targets[i, :seq_len] = item["targets"]
                attention_mask[i, :seq_len] = True

            out = {
                "input_ids": input_ids,
                "targets": targets,
                "attention_mask": attention_mask
            }
            if conditioning == "composition":
                cond_masks = []
                for i, item in enumerate(batch):
                    cond_masks.append(
                        build_composition_condition_mask(
                            item["input_ids"], data_token_id, newline_token_id
                        )
                    )
                condition_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
                for i, cm in enumerate(cond_masks):
                    condition_mask[i, : cm.size(0)] = cm
                out["condition_mask"] = condition_mask
            return out

        return collate_fn

    def train_dataloader(self):
        if self.train_dataset is None:
            self.setup()
        collate_fn = self._create_collate_fn()
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,  # prefetch batches
            collate_fn=collate_fn,
            drop_last=True,  # drop the final incomplete batch
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            self.setup()
        collate_fn = self._create_collate_fn()
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,  # prefetch batches
            collate_fn=collate_fn,
        )

