"""
generate_csp.py - Crystal Structure Prediction (CSP) sampling
=============================================================

Given a composition (reduced formula), sample several complete candidate
crystal structures in CIF form from the DMLM diffusion model, i.e.
p(CIF | composition).

Differences from unconditional generation (generate_dmlm.py):
- The start of the sequence is pinned to the exact composition prefix used in
  training (``data_<formula>\\n``).
- Every remaining position starts as [MASK] and is iteratively denoised into
  the structural part.
- The composition tokens keep their original ids throughout sampling, enforced
  twice: they are protected via ``partial_masks`` and force-restored after each
  denoising step.

Usage
-----
One composition, 10 candidates:
    python generate_csp.py --checkpoint <ckpt> --composition NaCl --num_samples 10

Several compositions (``num_samples`` candidates each):
    python generate_csp.py --checkpoint <ckpt> \\
        --composition NaCl --composition SiO2 --composition LiFePO4 \\
        --num_samples 5

A composition may also be written with the ``data_`` prefix (matching the first
line of the dataset):
    python generate_csp.py --checkpoint <ckpt> --composition data_NaCl

Output:
    <outdir>/NaCl/candidate_0.cif
    <outdir>/NaCl/candidate_1.cif
    ...
    <outdir>/SiO2/candidate_0.cif ...

Arguments:
    --checkpoint is a ``*.ckpt`` produced by training (see README for where to
    obtain one).

Environment: this script puts the repository's own ``src`` on ``sys.path``
itself, so no PYTHONPATH setup is needed.
"""

import argparse
import os
import sys

import torch

# Allow running straight from the project root without installing the package.
# The repository's own src is prepended unconditionally: a stale editable
# install of byprot elsewhere would otherwise shadow it and provide a version
# without the CSP sampling methods.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))

from byprot.models.dmlm.dmlm import DiffusionMaterialLanguageModel as DMLM


def parse_args():
    p = argparse.ArgumentParser(
        description="CSP: composition-conditioned crystal structure prediction sampling."
    )
    p.add_argument("--checkpoint", type=str, required=True,
                   help="path to a model checkpoint (a *.ckpt produced by train.py)")
    p.add_argument("--composition", type=str, action="append", default=None,
                   help="a composition; repeat the flag to pass several, e.g. "
                        "--composition NaCl --composition SiO2")
    p.add_argument("--compositions-file", type=str, default=None,
                   help="text file with one composition per line (alternative to --composition)")
    p.add_argument("--num-samples", type=int, default=5,
                   help="number of candidate structures to generate per composition")
    p.add_argument("--batch-size", type=int, default=4,
                   help="rows per sampling batch (rows may come from any composition)")
    p.add_argument("--max-iter", type=int, default=500,
                   help="number of iterative denoising steps")
    p.add_argument("--seq-len", type=int, default=500,
                   help="total sequence length (composition prefix + structure tokens to generate)")
    p.add_argument("--seed", type=int, default=0,
                   help="random seed (each batch uses seed + batch_index)")
    p.add_argument("--outdir", type=str, default="csp_output",
                   help="output directory: <outdir>/<composition>/candidate_<i>.cif")
    p.add_argument("--device", type=str, default=None,
                   help="cuda:0 / cpu (auto-detected by default)")
    return p.parse_args()


def load_model(checkpoint, device):
    print(f"\nLoading model from {checkpoint} ...")
    dmlm = DMLM.from_pretrained(checkpoint).to(device)
    dmlm.eval()
    print(f"Model loaded! vocab_size={dmlm.tokenizer.vocab_size}, "
          f"mask_id={dmlm.mask_id}, pad_id={dmlm.pad_id}")
    return dmlm


def encode_compositions(dmlm, compositions, seq_len):
    """Encode each composition into training-format token ids (``data_<formula>\\n``)."""
    encoded = []  # (composition, token_ids, token_strings)
    for comp in compositions:
        token_ids, token_strings = dmlm.tokenizer.encode_composition(comp, add_newline=True)
        if len(token_ids) > seq_len:
            raise ValueError(
                f"composition {comp!r} encodes to {len(token_ids)} tokens "
                f"({token_strings!r}), which exceeds seq_len={seq_len}!"
            )
        # Roundtrip check: decode(ids) must reproduce data_<comp>\n
        decoded = dmlm.tokenizer.decode(token_ids)
        expected = "data_" + comp.strip().removeprefix("data_") + "\n"
        if decoded != expected:
            raise ValueError(
                f"composition {comp!r} failed the roundtrip: decode={decoded!r}, expected={expected!r}"
            )
        encoded.append((comp, token_ids, token_strings))
        print(f"composition {comp!r}: tokens={token_strings!r} ids={token_ids} "
              f"(len={len(token_ids)}) decode='{decoded}'")
    return encoded


def build_chunk(rows, seq_len, mask_id, device):
    """rows: list of (row_meta, cond_ids); pack them into one input batch.

    Condition lengths differ across compositions, hence the per-row
    condition_mask [B, L].
    """
    bsz = len(rows)
    input_tokens = torch.full((bsz, seq_len), mask_id, dtype=torch.long, device=device)
    condition_mask = torch.zeros((bsz, seq_len), dtype=torch.bool, device=device)
    for i, (_, cond_ids, _) in enumerate(rows):
        k = len(cond_ids)
        input_tokens[i, :k] = torch.tensor(cond_ids, dtype=torch.long, device=device)
        condition_mask[i, :k] = True
    return input_tokens, condition_mask


def decode_row(tokenizer, tokens, pad_strs=("<pad>", "<eos>", "<bos>")):
    """Decode one token row, truncating at the first pad/eos/bos special token and stripping."""
    text = tokenizer.decode(tokens)
    for ps in pad_strs:
        if ps in text:
            text = text[: text.index(ps)]
    return text.strip()


def main():
    args = parse_args()

    compositions = list(args.composition) if args.composition else []
    if args.compositions_file:
        with open(args.compositions_file, "r", encoding="utf-8") as f:
            compositions += [ln.strip() for ln in f if ln.strip()]
    if not compositions:
        raise SystemExit("Please pass --composition (repeatable) or --compositions-file!")
    # Deduplicate while preserving order. The original list is consumed here
    # into a new one, so build `deduped` separately rather than reusing the name.
    deduped, seen = [], set()
    for c in compositions:
        key = c.strip().removeprefix("data_")
        if key not in seen:
            seen.add(key)
            deduped.append(c)
    compositions = deduped

    if args.device is not None:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda:0"
        if device == "cuda:0":
            torch.cuda.set_device(0)
    else:
        device = "cpu"
    print(f"Using device: {device}")

    dmlm = load_model(args.checkpoint, device)
    mask_id = dmlm.mask_id
    if mask_id is None:
        raise SystemExit("Model has no mask token id; cannot initialize a [MASK] sequence!")

    encoded = encode_compositions(dmlm, compositions, args.seq_len)

    # num_samples rows per composition
    rows = []
    for comp, cond_ids, token_strings in encoded:
        for s in range(args.num_samples):
            rows.append((comp, cond_ids, token_strings))

    os.makedirs(args.outdir, exist_ok=True)
    total_saved = 0
    saved_counter = {comp: 0 for comp, _, _ in encoded}  # saved count per composition

    # Sample batch by batch; rows within a batch may belong to different
    # compositions, since the per-row condition length is variable.
    for start in range(0, len(rows), args.batch_size):
        chunk = rows[start : start + args.batch_size]
        input_tokens, condition_mask = build_chunk(
            chunk, args.seq_len, mask_id, device
        )

        # RNG seed (makes a given batch reproducible)
        torch.manual_seed(args.seed + start)

        with torch.no_grad():
            samples = dmlm.generate(
                input_tokens=input_tokens,
                max_iter=args.max_iter,
                partial_masks=condition_mask,
                condition_mask=condition_mask,
                condition_ids=input_tokens,
            )
        samples = samples.detach().cpu()

        # Validate, then save
        for i, (comp, cond_ids, token_strings) in enumerate(chunk):
            row_tokens = samples[i].tolist()
            k = len(cond_ids)
            # Critical assertion: the condition tokens must survive the whole
            # diffusion sampling loop unchanged.
            cond_ok = row_tokens[:k] == cond_ids
            if not cond_ok:
                print(f"[ERROR] the condition prefix of composition {comp!r} changed during sampling!")
                print(f"        expected={cond_ids}")
                print(f"        got     ={row_tokens[:k]}")
                raise RuntimeError(
                    "condition tokens were modified during sampling!"
                )

            text = decode_row(dmlm.tokenizer, row_tokens)
            comp_dir = os.path.join(args.outdir, comp.strip().removeprefix("data_"))
            os.makedirs(comp_dir, exist_ok=True)
            idx_in_comp = saved_counter[comp]
            saved_counter[comp] += 1
            save_path = os.path.join(comp_dir, f"candidate_{idx_in_comp}.cif")
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(text + "\n")
            total_saved += 1

            if start == 0 and i == 0:
                print("\n[CSP debug] preview of the first sample (first 12 lines):")
                for ln in text.split("\n")[:12]:
                    print("   " + ln)
                print("   ...")
            print(f"  saved {save_path}  (cond={len(cond_ids)} tokens, "
                  f"{len(row_tokens)} tokens total)")

    print(f"\nDone: generated {total_saved} candidate CIFs into {args.outdir}")
    for comp, cond_ids, _ in encoded:
        key = comp.strip().removeprefix("data_")
        comp_dir = os.path.join(args.outdir, key)
        print(f"  - {key}: {comp_dir}/candidate_0.cif ... candidate_{args.num_samples - 1}.cif")


if __name__ == "__main__":
    main()
