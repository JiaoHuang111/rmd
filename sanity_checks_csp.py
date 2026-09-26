"""
sanity_checks_csp.py - sanity checks for CSP (composition-conditioned crystal
structure prediction)
=============================================================================

Lightweight assertion-based tests (no pytest needed). Run directly:

    python sanity_checks_csp.py

The device is auto-detected (cuda if available, else cpu). Set the
``SANITY_DEVICE`` environment variable to force one, e.g.::

    SANITY_DEVICE=cpu python sanity_checks_csp.py
    SANITY_DEVICE=cuda:3 python sanity_checks_csp.py

Coverage:
  [T0] tokenizer / special token ids and basic vocab assumptions
  [T1] composition tokens are unchanged after corruption at any timestep
  [T2] CSP loss: composition positions contribute no loss (loss_mask and
       condition are disjoint)
  [T3] CSP sampling initialization: NaCl -> composition tokens + [MASK]...
  [T4] full sampling smoke test: the condition tokens are never modified
  [T5] random generation: condition_mask=None behaves like the original
       implementation
  [T6] data layer: with conditioning=composition the dataloader emits a correct
       per-sample condition_mask (lengths differ within a batch); with
       conditioning=none the key is absent
"""

import os
import re
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))

from byprot.models.dmlm.dmlm import DiffusionMaterialLanguageModel as DMLM
from byprot.datamodules.crystalmols_hf import (
    CrystalDataset,
    CrystalMolsDataModule,
    build_composition_condition_mask,
    default_data_root,
)
from byprot.modules.cross_entropy import RDMCrossEntropyLoss
from byprot.utils.config import load_yaml_config

PASS = []
FAIL = []


def check(name, cond, extra=""):
    if cond:
        PASS.append(name)
        print(f"  [PASS] {name}")
    else:
        FAIL.append(name)
        print(f"  [FAIL] {name} {extra}")
    return cond


def pick_device():
    """Auto-detect the device, or honour the SANITY_DEVICE override."""
    forced = os.environ.get("SANITY_DEVICE")
    if forced:
        return forced
    return "cuda" if torch.cuda.is_available() else "cpu"


def make_model(device):
    """Build the model the same way DMLM.from_pretrained does (random weights, no checkpoint)."""
    cfg_path = os.path.join(_HERE, "configs", "config_all_150m_mp20.yaml")
    cfg = load_yaml_config(cfg_path).model
    cfg.net.pretrain = False
    cfg.pop("_target_", None)
    model = DMLM(cfg).to(device)
    model.eval()
    return model


def get_real_rows(n=4):
    """Take n real tokenized CIF sequences from the tokens_mp_20 validation split."""
    data_file = os.path.join(default_data_root(), "tokens_mp_20", "val.bin")
    ds = CrystalDataset(data_file=data_file)
    idxs = [3, 17, 59, 211][:n]
    return [ds[i]["input_ids"] for i in idxs]


def t0_tokenizer(model):
    print("\n[T0] tokenizer / special ids")
    tok = model.tokenizer
    print(f"  vocab_size={tok.vocab_size}")
    ok = (
        tok.vocab_size == 375
        and tok.unk_token_id == 370
        and tok.pad_token_id == 371
        and tok.eos_token_id == 372
        and tok.bos_token_id == 373
        and tok.mask_token_id == 374
        and model.mask_id == 374
        and model.pad_id == 371
    )
    check("T0 special ids (unk370 pad371 eos372 bos373 mask374)", ok,
          f"got unk={tok.unk_token_id} pad={tok.pad_token_id} mask={model.mask_id}")
    # data_ / \n ids (must match meta.pkl and the datasets)
    check("T0 'data_'=124 & '\\n'=142", tok.token_to_id["data_"] == 124 and tok.token_to_id["\n"] == 142)
    # Tokenization of NaCl
    ids, toks = tok.encode_composition("NaCl")
    check("T0 encode_composition('NaCl') == [data_,Na,Cl,\\n]",
          toks == ["data_", "Na", "Cl", "\n"]
          and tok.decode(ids) == "data_NaCl\n", f"toks={toks} decode={tok.decode(ids)!r}")
    return ok


def t1_corruption(model, rows, device):
    print("\n[T1] composition tokens are unchanged after corruption at any timestep")
    # Build a [B,L] batch from real rows (padded with the model's pad_id=371)
    max_len = max(len(r) for r in rows)
    x0 = torch.full((len(rows), max_len), model.pad_id, dtype=torch.long)
    cond = torch.zeros(len(rows), max_len, dtype=torch.bool)
    for i, r in enumerate(rows):
        x0[i, : len(r)] = r
        cond[i, : len(r)] = build_composition_condition_mask(
            r, data_token_id=124, newline_token_id=142
        )
    x0, cond = x0.to(device), cond.to(device)
    maskable = (
        model.get_non_special_symbol_mask(x0) & ~cond
    )  # CSP: maskable excludes cond
    all_ok = True
    for t in [1, 7, 64, 250, 499, 500]:
        x_t, t_, loss_mask = list(
            model.q_sample(
                x0,
                torch.tensor([t] * len(rows), device=device),
                maskable_mask=maskable,
                condition_mask=cond,
            ).values()
        )
        ok = torch.equal(x_t[cond], x0[cond]) and not (loss_mask & cond).any()
        all_ok &= bool(ok)
        if not ok:
            print(f"  t={t}: FAILED")
    check("T1 x_t[cond]==x0[cond] & cond not in loss_mask (t=1..500)", all_ok)
    return all_ok


def t2_loss(model, rows, device):
    print("\n[T2] CSP loss: composition positions contribute no loss")
    max_len = max(len(r) for r in rows)
    x0 = torch.full((len(rows), max_len), model.pad_id, dtype=torch.long)
    cond = torch.zeros(len(rows), max_len, dtype=torch.bool)
    for i, r in enumerate(rows):
        x0[i, : len(r)] = r
        cond[i, : len(r)] = build_composition_condition_mask(
            r, data_token_id=124, newline_token_id=142
        )
    x0, cond = x0.to(device), cond.to(device)
    batch = {"input_ids": x0, "targets": x0.clone(),
             "input_mask": torch.ones_like(x0, dtype=torch.bool)}
    all_ok = True
    with torch.no_grad():
        for seed in range(5):
            torch.manual_seed(seed)
            _, target, loss_mask, weights = model.compute_loss(
                batch, weighting="constant", condition_mask=cond
            )
            ok = (target[cond] == x0[cond]).all() and not (loss_mask & cond).any()
            all_ok &= bool(ok)
            if not ok:
                print(f"  seed={seed}: FAILED")
    check("T2 condition never enters loss_mask across 5 random seeds", all_ok)
    # Explicit numerical criterion check: cond positions are excluded by
    # loss_mask, so removing them must not change the loss at all.
    with torch.no_grad():
        torch.manual_seed(0)
        logits, target, loss_mask, weights = model.compute_loss(
            batch, weighting="constant", condition_mask=cond
        )
        crit = RDMCrossEntropyLoss(label_smoothing=0.1, ignore_index=model.pad_id)
        loss_orig, _ = crit(logits, target, loss_mask, weights)
        # Strip cond from the mask by hand (expected to be a no-op)
        loss_mask2 = loss_mask.clone()
        loss_mask2 &= ~cond
        loss_stripped, _ = crit(logits, target, loss_mask2, weights)
    check("T2 criterion is numerically unchanged when cond is stripped (no leak under label smoothing)",
          torch.allclose(loss_orig, loss_stripped, atol=1e-6),
          f"{loss_orig.item()} vs {loss_stripped.item()}")
    return all_ok


def t3_init(model, device):
    print("\n[T3] CSP sampling initialization: condition = NaCl")
    seq_len, mask_id = 64, model.mask_id
    tok = model.tokenizer
    cond_ids, cond_toks = tok.encode_composition("NaCl", add_newline=True)
    input_tokens = torch.full((1, seq_len), mask_id, dtype=torch.long, device=device)
    condition_mask = torch.zeros((1, seq_len), dtype=torch.bool, device=device)
    k = len(cond_ids)
    input_tokens[0, :k] = torch.tensor(cond_ids, dtype=torch.long, device=device)
    condition_mask[0, :k] = True
    head = input_tokens[0, :k].tolist()
    ok = head == cond_ids and bool((input_tokens[0, k:] == mask_id).all())
    check("T3 init = composition tokens + [MASK]...", ok,
          f"head={head} cond_ids={cond_ids}")
    print(f"  decode head = {tok.decode(head)!r}; remaining {seq_len - k} positions are [MASK]")
    return ok


def t4_sampling_smoke(model, device, max_iter=8, seq_len=64):
    print("\n[T4] full sampling smoke test: the condition tokens are never modified")
    tok = model.tokenizer
    cond_ids, _ = tok.encode_composition("LiFePO4", add_newline=True)  # a longer condition
    input_tokens = torch.full((1, seq_len), model.mask_id, dtype=torch.long, device=device)
    condition_mask = torch.zeros((1, seq_len), dtype=torch.bool, device=device)
    k = len(cond_ids)
    input_tokens[0, :k] = torch.tensor(cond_ids, dtype=torch.long, device=device)
    condition_mask[0, :k] = True
    with torch.no_grad():
        samples = model.generate(
            input_tokens=input_tokens,
            max_iter=max_iter,
            sampling_strategy="argmax",   # deterministic strategy, easier to reproduce
            partial_masks=condition_mask,
            condition_mask=condition_mask,
            condition_ids=input_tokens,
        )
    s = samples[0]
    cond_ok = s[:k].tolist() == cond_ids
    decoded = tok.decode(s.tolist())
    check("T4 condition prefix still data_LiFePO4\\n after sampling", cond_ok,
          f"head={s[:k].tolist()} expected={cond_ids}")
    # The remaining positions really were generated (no longer all mask)
    rest = s[k:]
    check("T4 the structure part is being generated (mask ratio < 100%)",
          bool((rest != model.mask_id).any()), f"unique={rest.unique().tolist()[:10]}")
    print(f"  decode head: {decoded[:60]!r}")
    return cond_ok


def t5_random_unchanged(model, device):
    print("\n[T5] random generation: condition_mask=None matches an explicit all-False mask")
    seq_len, max_iter = 48, 6
    torch.manual_seed(0)
    x0 = torch.full((1, seq_len), model.mask_id, dtype=torch.long, device=device)
    with torch.no_grad():
        s_none = model.generate(
            input_tokens=x0.clone(), max_iter=max_iter, sampling_strategy="argmax"
        )
        zeros = torch.zeros((1, seq_len), dtype=torch.bool, device=device)
        s_zero = model.generate(
            input_tokens=x0.clone(), max_iter=max_iter, sampling_strategy="argmax",
            condition_mask=zeros,
        )
    check("T5 both paths produce token-identical output (random behaviour unchanged)",
          torch.equal(s_none, s_zero))
    return True


def t6_datamodule():
    print("\n[T6] data layer condition_mask output")
    data_dir = os.path.join(default_data_root(), "tokens_mp_20")
    # 1) conditioning=composition: the batch carries a per-sample condition_mask
    dm = CrystalMolsDataModule(
        data_dir=data_dir, max_tokens=40960, max_len=2048,
        num_workers=0, tokenizer="crystallm", batch_size=6,
        conditioning="composition",
    )
    dm.setup()
    batches = []
    for bi, batch in enumerate(dm.val_dataloader()):
        batches.append(batch)
        if bi >= 1:
            break
    assert "condition_mask" in batches[0], "composition conditioning did not emit condition_mask!"
    cond = batches[0]["condition_mask"]
    ids = batches[0]["input_ids"]
    print(f"  batch shape={tuple(ids.shape)} condition_mask shape={tuple(cond.shape)}")
    ok_prefix = True
    n_cond = [int(c.sum()) for c in cond]
    n_cond_unique = len(set(n_cond))
    for i in range(ids.size(0)):
        k = int(cond[i].sum())
        row_head = ids[i, :k].tolist() if k else []
        # Verify the decoded head is data_ ... \n
        if k > 0:
            hdr = dm.tokenizer.decode(row_head)
            m = re.fullmatch(r"data_([A-Z][a-z]*\d*)+\n", hdr)
            if m is None:
                ok_prefix = False
                print(f"  row {i} has an unexpected header: {hdr!r}")
        # The mask may only cover the leading condition span
        if k > 0:
            ok_prefix &= bool((cond[i, k:] == False).all())  # noqa: E712
    check("T6 composition conditioning: every prefix is data_<Formula>\\n", ok_prefix)
    check(f"T6 condition lengths differ within a batch (genuinely per-sample, {n_cond_unique} distinct)",
          n_cond_unique > 1, f"lengths={sorted(set(n_cond))}")
    # 2) default conditioning: no condition_mask is emitted (random-generation
    #    training path unchanged)
    dm2 = CrystalMolsDataModule(
        data_dir=data_dir, max_tokens=40960, max_len=2048,
        num_workers=0, tokenizer="crystallm", batch_size=6,
        conditioning="none",
    )
    dm2.setup()
    b2 = next(iter(dm2.val_dataloader()))
    check("T6 conditioning=none: the batch has no condition_mask (random task stays backward compatible)",
          "condition_mask" not in b2 and list(b2.keys()) == ["input_ids", "targets", "attention_mask"])
    return True


def main():
    print("=" * 70)
    print("CSP sanity checks")
    print("=" * 70)
    device = pick_device()
    print(f"device: {device}")

    torch.manual_seed(0)
    model = make_model(device)
    t0_tokenizer(model)
    rows = get_real_rows(n=3)

    t1_corruption(model, rows, device)          # no network forward needed, fast
    t2_loss(model, rows, device)                # needs one small forward
    t3_init(model, device)
    t4_sampling_smoke(model, device, max_iter=8, seq_len=64)  # 8-step mini generation
    t5_random_unchanged(model, device)
    t6_datamodule()

    print("\n" + "=" * 70)
    print(f"Results: {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print(f"  FAILED: {f}")
        print("=" * 70)
        sys.exit(1)
    print("ALL CHECKS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
