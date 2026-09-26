"""
generate_candidates.py — candidate generation for benchmarking (DMLM-CSP, p(CIF|composition)).

For every prompt (composition) in the manifest, generates `--shots` candidate structures
independently, named `<outdir>/<id>__<k>.cif` following the CrystaLLM convention
(k=0..shots-1, k=0 being the 1-shot "first generation"). The generation logic is the same
as in generate_csp.py:
  - sequence = fixed condition prefix (data_<formula>\\n, same tokenization as the training
    rows) + a length-budget number of [MASK] tokens;
  - DMLM diffusion sampling for --max-iter steps; the condition and the pad tail beyond the
    budget stay fixed;
  - decode is truncated at the first <pad>/<eos>/<bos> (same as generate_csp.decode_row).

Length budget (fixed per prompt, deterministic; produced by build_budgets.py, see
--budgets-json):
  budget = median(original token length of the train/val rows with the same reduced
  composition) - cond_len + slack(8), clamped to [50, max_total-cond]; for prompts whose
  composition is missing from the training corpus (compositions unique to test), falls back
  to the dataset-level global median length and records that (budget_meta.json).
Fixed-length diffusion cannot stop on its own, so anchoring on "the model's own prior length
for that composition" is a substitute for the CrystaLLM convention in which the
autoregressive model decides where to stop from its own prior (unrelated to CrystaLLM's
official --num-gens semantics; all 20 candidates are sampled independently).

Randomness: seed = --seed + global row index (row index = the prompt's position in the
manifest x shots + k), reset per batch to the seed of the batch's first row -- tied to the
batch split / batch size, fixed and recorded for full runs.
"""
import argparse
import json
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
# Make sure the byprot imported below is the one in this checkout: a stale
# editable install elsewhere could otherwise take precedence.
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "src"))

from byprot.models.dmlm.dmlm import DiffusionMaterialLanguageModel as DMLM  # noqa: E402


def load_model(checkpoint, device):
    print(f"Loading model from {checkpoint} ...", flush=True)
    dmlm = DMLM.from_pretrained(checkpoint).to(device)
    dmlm.eval()
    print(f"Model loaded! vocab_size={dmlm.tokenizer.vocab_size}, "
          f"mask_id={dmlm.mask_id}, pad_id={dmlm.pad_id}", flush=True)
    return dmlm


def cond_from_prompt_text(dmlm, prompt_text):
    """prompt file content (data_<formula>\\n) -> condition token ids (strict encoding via the wrapper)."""
    text = prompt_text.strip()
    assert text.startswith("data_"), f"bad prompt text {prompt_text!r}"
    formula = text[len("data_"):]
    token_ids, token_strings = dmlm.tokenizer.encode_composition(formula, add_newline=True)
    decoded = dmlm.tokenizer.decode(token_ids)
    expected = "data_" + formula + "\n"
    if decoded != expected:
        raise ValueError(f"roundtrip fail: decode={decoded!r} expected={expected!r}")
    return token_ids


def build_chunk(rows, Lmax, mask_id, pad_id, device):
    """condition prefix + [MASK] span (each row has length total) + pad tail;
    condition_mask protects cond+pad."""
    bsz = len(rows)
    input_tokens = torch.full((bsz, Lmax), pad_id, dtype=torch.long, device=device)
    condition_mask = torch.zeros((bsz, Lmax), dtype=torch.bool, device=device)
    for i, (_, _, cond_ids, total) in enumerate(rows):
        k = len(cond_ids)
        input_tokens[i, :k] = torch.tensor(cond_ids, dtype=torch.long, device=device)
        input_tokens[i, k:total] = mask_id
        condition_mask[i, :k] = True
        condition_mask[i, total:] = True  # pad tail is fixed, not denoised
    return input_tokens, condition_mask


def decode_row(tokenizer, tokens, pad_strs=("<pad>", "<eos>", "<bos>")):
    text = tokenizer.decode(tokens)
    for ps in pad_strs:
        if ps in text:
            text = text[: text.index(ps)]
    return text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompts-dir", required=True, help="prompts/<ds> directory")
    ap.add_argument("--budgets-json", required=True, help="{id: budget_tokens}")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--shots", type=int, default=20)
    ap.add_argument("--max-iter", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--start-idx", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="0 = all prompts")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-total", type=int, default=1010,
                    help="upper bound on cond+budget (ESM position emb 1026 minus pad margin)")
    args = ap.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda"):
        torch.cuda.set_device(int(device.split(":")[1]))
    print(f"Using device: {device}", flush=True)

    dmlm = load_model(args.ckpt, device)
    mask_id, pad_id = dmlm.mask_id, dmlm.pad_id
    if mask_id is None or pad_id is None:
        raise SystemExit("model is missing mask/pad token id!")

    budgets = json.load(open(args.budgets_json))
    prompt_ids = sorted(b for b in budgets)  # manifest order = key order of the budgets file
    # Kept consistent with the order of the budgets file: file key order is fine
    # (deterministic as long as the sorted keys match those at build time)
    if args.limit:
        prompt_ids = prompt_ids[args.start_idx: args.start_idx + args.limit]
    else:
        prompt_ids = prompt_ids[args.start_idx:]
    print(f"{len(prompt_ids)} prompts, {args.shots} shots each -> "
          f"{len(prompt_ids) * args.shots} candidates", flush=True)

    prompts = [(pid, open(os.path.join(args.prompts_dir, f"{pid}.txt"), encoding="utf-8").read())
               for pid in prompt_ids]

    # ---- generation (same batch sampling logic as generate_csp.py; row lengths vary, aligned by pad) ----
    # rows_all: manifest order x shots, row index = pid_idx * shots + k
    rows_all = []
    for pid_idx, (pid, ptxt) in enumerate(prompts):
        cond = cond_from_prompt_text(dmlm, ptxt)
        for k in range(args.shots):
            rows_all.append((pid, pid_idx, cond, len(cond) + budgets[pid]))

    os.makedirs(args.outdir, exist_ok=True)
    n_saved = 0
    t0 = time.time()
    n_batches = (len(rows_all) + args.batch_size - 1) // args.batch_size
    for b in range(n_batches):
        chunk = rows_all[b * args.batch_size: (b + 1) * args.batch_size]
        global_idx0 = b * args.batch_size
        Lmax = max(t for _, _, _, t in chunk)
        input_tokens, condition_mask = build_chunk(
            chunk, Lmax, mask_id, pad_id, device)

        torch.manual_seed(args.seed + global_idx0)  # tied to the batch split (on record)

        with torch.no_grad():
            samples = dmlm.generate(
                input_tokens=input_tokens,
                max_iter=args.max_iter,
                partial_masks=condition_mask,
                condition_mask=condition_mask,
                condition_ids=input_tokens,
            )
        samples = samples.detach().cpu()

        for i, (pid, pid_idx, cond_ids, total) in enumerate(chunk):
            row_tokens = samples[i].tolist()
            k = len(cond_ids)
            cond_ok = row_tokens[:k] == cond_ids
            if not cond_ok:
                raise RuntimeError(f"{pid}: condition tokens modified during sampling!")
            text = decode_row(dmlm.tokenizer, row_tokens)
            kk = global_idx0 + i - pid_idx * args.shots
            path = os.path.join(args.outdir, f"{pid}__{kk}.cif")
            with open(path, "w", encoding="utf-8") as f:
                f.write(text + "\n")
            n_saved += 1

        if b % 25 == 0:
            el = time.time() - t0
            rate = n_saved / el if el else 0
            print(f"  ... batch {b}/{n_batches}: saved={n_saved} "
                  f"({rate:.1f} cand/s, elapsed {el:.0f}s)", flush=True)

    print(f"✓ done: {n_saved} candidates -> {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
