"""
generate_gtmask.py — CSP validation/test generation (user-specified protocol: fixed-length
GT-mask initialisation).

Protocol (matching the constraint that "the model is particularly sensitive to the initial
sequence length"):
  1. Take the GT test data (the canonical rows produced by prepare_gt.py = the token sequence
     of the GT CIF row under the official CrystaLLM chain);
  2. The composition information (the tokens of the first line `data_<formula>\n`) is **left
     unchanged**;
  3. All other positions are set to [MASK];
  4. The total sequence length = the GT row's original token count (same length before and
     after masking, matching the "denoise the whole row" shape used during training and
     validation).

Generation: DMLM diffusion sampling for --max-iter steps, with the composition positions
protected throughout (partial_masks/condition_mask force the condition tokens back at every
step); outputs `<outdir>/<id>__<k>.cif` (k=0 is the 1-shot first generation).

Randomness: per batch `torch.manual_seed(seed + global start row index of the batch)`,
row index = id ordinal x shots + k (same as generate_candidates.py, which makes it easy to
compare against the old-protocol results).
"""
import argparse
import gzip
import json
import os
import pickle
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "src"))
from _paths import tokens_dir  # noqa: E402

import importlib.util
_spec = importlib.util.spec_from_file_location("gc", os.path.join(_HERE, "generate_candidates.py"))
gc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ds", default="mp_20")
    ap.add_argument("--gt-tokens", default=None, help="gt_rows_tokens.pkl.gz (default data/<ds>/gt_rows_tokens.pkl.gz)")
    ap.add_argument("--gt-meta", default=None, help="gt_rows.json (default data/<ds>/gt_rows.json)")
    ap.add_argument("--tokens-dir", default=None,
                    help="directory holding the corpus meta.pkl (default data-bin/crystalmols/tokens_<ds>)")
    ap.add_argument("--ids-json", default=None, help="optional: run only these ids (JSON list)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--shots", type=int, default=20)
    ap.add_argument("--max-iter", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--start-idx", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-total", type=int, default=1010)
    args = ap.parse_args()

    data_dir = os.path.join(_HERE, "data", args.ds)
    gt_tokens_path = args.gt_tokens or os.path.join(data_dir, "gt_rows_tokens.pkl.gz")
    gt_meta_path = args.gt_meta or os.path.join(data_dir, "gt_rows.json")
    corpus_dir = args.tokens_dir or tokens_dir(args.ds)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda"):
        torch.cuda.set_device(int(device.split(":")[1]))
    print(f"device={device}  ds={args.ds}", flush=True)

    with gzip.open(gt_tokens_path, "rb") as f:
        gt_tokens = pickle.load(f)
    gt_meta = json.load(open(gt_meta_path))
    meta = pickle.load(open(os.path.join(corpus_dir, "meta.pkl"), "rb"))
    stoi = meta["stoi"]

    dmlm = gc.load_model(args.ckpt, device)
    mask_id, pad_id = dmlm.mask_id, dmlm.pad_id

    ids = sorted(gt_tokens)
    if args.ids_json:
        wanted = set(json.load(open(args.ids_json)))
        ids = [i for i in ids if i in wanted]
    ids = ids[args.start_idx: args.start_idx + args.limit] if args.limit else ids[args.start_idx:]
    print(f"{len(ids)} ids × {args.shots} shots = {len(ids)*args.shots} candidates", flush=True)

    rows_all, row_meta = [], []
    for pid_idx, pid in enumerate(ids):
        toks = gt_tokens[pid]
        assert all(t in stoi for t in toks), f"{pid}: token not in the vocabulary"
        tids = [stoi[t] for t in toks]
        n = len(tids)
        cond_len = gt_meta[pid]["cond_len"]
        if n > args.max_total:
            print(f"  [warn] {pid}: row_len={n} > max_total={args.max_total}, truncating", flush=True)
            tids = tids[:args.max_total]
            n = args.max_total
        row_meta.append((pid, cond_len, n, gt_meta[pid]["row_len"]))
        for k in range(args.shots):
            rows_all.append((pid, pid_idx, tids[:cond_len], n))
    # save the rows used for generation (token string ids) for auditing
    np_meta = {f"{pid}|{pid_idx}": {"cond_len": cl, "total": n, "gt_row_len": rl}
               for pid, pid_idx, cl, n, rl in [(p, i, c, n, r) for i, (p, c, n, r) in enumerate(row_meta)]}
    os.makedirs(args.outdir, exist_ok=True)
    json.dump(np_meta, open(os.path.join(args.outdir, "_init_meta.json"), "w"), indent=1)

    n_saved, idx = 0, 0
    t0 = time.time()
    n_batches = (len(rows_all) + args.batch_size - 1) // args.batch_size
    for b in range(n_batches):
        chunk_tuples = rows_all[b * args.batch_size: (b + 1) * args.batch_size]
        # build_chunk expects (pid, pid_idx, cond_ids, total)
        chunk = list(chunk_tuples)  # (pid, pid_idx, cond_ids(list), total)
        global_idx0 = b * args.batch_size
        Lmax = max(t for _, _, _, t in chunk)
        input_tokens, condition_mask = gc.build_chunk(chunk, Lmax, mask_id, pad_id, device)

        torch.manual_seed(args.seed + global_idx0)
        with torch.no_grad():
            samples = dmlm.generate(
                input_tokens=input_tokens, max_iter=args.max_iter,
                partial_masks=condition_mask, condition_mask=condition_mask,
                condition_ids=input_tokens,
            )
        samples = samples.detach().cpu()

        for i, (pid, pid_idx, cond_ids, total) in enumerate(chunk_tuples):
            row = samples[i].tolist()
            k = len(cond_ids)
            if row[:k] != list(cond_ids):
                raise RuntimeError(f"{pid}: condition tokens modified during sampling!")
            text = gc.decode_row(dmlm.tokenizer, row)
            kk = global_idx0 + i - pid_idx * args.shots
            with open(os.path.join(args.outdir, f"{pid}__{kk}.cif"), "w", encoding="utf-8") as f:
                f.write(text + "\n")
            n_saved += 1
        if b % 25 == 0:
            el = time.time() - t0
            print(f"  ... batch {b}/{n_batches}: saved={n_saved} ({n_saved/max(el,1e-9):.1f} cand/s, {el:.0f}s)",
                  flush=True)

    print(f"✓ done: {n_saved} candidates -> {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
