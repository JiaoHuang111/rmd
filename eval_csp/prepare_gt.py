"""
prepare_gt.py — builds "official-definition" GT rows (producing both the GT structure files
and the token rows).

Official CrystaLLM chain (bin/prepare_csv_benchmark.py + bin/preprocess.py +
bin/tokenize_cifs.py):
  csv cif text
    --(1)--> Structure.from_str(cif) --(2)--> CifWriter(struct, symprec=0.1)   # standardized writing
    --(3)--> replace_data_formula_with_nonreduced_formula / semisymmetrize_cif /
             add_atomic_props_block / round_numbers(4)                          # preprocess.py
    --(4)--> strip each line, drop empty / '#' / pymatgen lines,
             append "\n" at the end                                             # tokenize_cifs.preprocess
    --(5)--> CIFTokenizer.tokenize_cif -> token string (no eos; ends with '\n','\n')

Output (per dataset):
  data/<ds>/gt_prep/<id>.cif   -- the standardized CIF text (= the GT structure under the
                                  official true-cifs definition)
  data/<ds>/gt_rows.json       -- {id: {row_len, cond_len, cond_text, n_unk, failed}}
  data/<ds>/gt_rows_tokens.pkl.gz -- {id: [token strings]} (used directly by the generation
                                  scripts)

Check: for every id, the data_ line (token string) of the canonical row should be
character-for-character identical to <id>.txt from the official prompt tars; mismatching ids
are counted and listed (to judge whether the chain is faithful).
"""
import argparse
import csv
import gzip
import importlib.util
import json
import os
import pickle
import re
import sys
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _paths import crystallm_csv  # noqa: E402


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


pt = load("preprocess_test", os.path.join(HERE, "preprocess_test.py"))
TK = load("cif_tokenizer", os.path.join(HERE, "..", "src", "byprot", "crystallm", "_tokenizer.py"))
tok = TK.CIFTokenizer()


def line_clean(s):
    ls = [l.strip() for l in s.split("\n")
          if len(l.strip()) > 0 and not l.strip().startswith("#") and "pymatgen" not in l]
    ls.append("\n")
    return "\n".join(ls)


def prepare_one(pid, csv_cif, pymatgen_version_ok=True):
    """official chain -> (tokens, gt_prep_text, err)"""
    from pymatgen.io.cif import CifWriter, Structure
    try:
        struct = Structure.from_str(csv_cif, fmt="cif")
        prepared = str(CifWriter(struct=struct, symprec=0.1))
    except Exception as e:
        return None, None, f"standardize: {type(e).__name__}: {e}"
    try:
        can = pt.augment_cif(pid, prepared)
    except Exception as e:
        return None, None, f"preprocess: {type(e).__name__}: {e}"
    toks = tok.tokenize_cif(line_clean(can))
    if not toks:
        return None, prepared, "empty tokenization"
    return toks, prepared, None


def _work(item):
    pid, cif = item
    toks, prepared, err = prepare_one(pid, cif)
    return pid, toks, prepared, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="mp_20")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--out-suffix", default="", help="output directory suffix (empty = formal; _smoke = trial)")
    args = ap.parse_args()

    csv_path = crystallm_csv(args.ds, "test")
    prompts_dir = os.path.join(HERE, "prompts", args.ds)
    out_dir = os.path.join(HERE, "data", args.ds, "gt_prep" + args.out_suffix)
    os.makedirs(out_dir, exist_ok=True)

    rows = {r["material_id"]: r["cif"] for r in csv.DictReader(open(csv_path))}
    pids = sorted(f[:-4] for f in os.listdir(prompts_dir) if f.endswith(".txt"))
    if args.limit:
        pids = pids[:args.limit]
    print(f"{args.ds}: {len(pids)} prompts (csv rows {len(rows)})", flush=True)

    import multiprocessing as mp
    items = [(pid, rows[pid]) for pid in pids if pid in rows]

    results = {}
    with mp.Pool(args.workers) as pool:
        for i, (pid, toks, prepared, err) in enumerate(pool.imap_unordered(_work, items, chunksize=16)):
            results[pid] = (toks, prepared, err)
            if (i + 1) % 500 == 0:
                print(f"  ... {i+1}/{len(items)}", flush=True)

    rows_json, tokens_pkl, n_ok, n_fail, n_cond_mismatch, unk_ids = {}, {}, 0, 0, 0, []
    cond_mismatch_ids = []
    for pid in sorted(results):
        toks, prepared, err = results[pid]
        prompt = open(os.path.join(prompts_dir, f"{pid}.txt")).read()
        if toks is None:
            rows_json[pid] = {"failed": err}
            n_fail += 1
            continue
        with open(os.path.join(out_dir, f"{pid}.cif"), "w") as f:
            f.write(prepared if prepared else "")
        try:
            nl = toks.index("\n")
        except ValueError:
            nl = len(toks)
        cond = "".join(toks[:nl + 1])
        n_unk = sum(1 for t in toks if t == "<unk>")
        if n_unk:
            unk_ids.append(pid)
        rows_json[pid] = {"row_len": len(toks), "cond_len": nl + 1, "cond_text": cond,
                          "n_unk": n_unk, "cond_matches_prompt": cond == prompt}
        tokens_pkl[pid] = toks
        n_ok += 1
        if cond != prompt:
            n_cond_mismatch += 1
            if len(cond_mismatch_ids) < 10:
                cond_mismatch_ids.append((pid, cond.strip(), prompt.strip()))

    json.dump(rows_json, open(os.path.join(HERE, "data", args.ds, f"gt_rows{args.out_suffix}.json"), "w"), indent=1)
    with gzip.open(os.path.join(HERE, "data", args.ds, f"gt_rows_tokens{args.out_suffix}.pkl.gz"), "wb") as f:
        pickle.dump(tokens_pkl, f, protocol=pickle.HIGHEST_PROTOCOL)

    import numpy as np
    lens = [v["row_len"] for v in rows_json.values() if "row_len" in v]
    print(f"\n=== {args.ds} results ===")
    print(f"ok={n_ok} failed={n_fail}  cond!=prompt={n_cond_mismatch}  unk_rows={len(unk_ids)}")
    if lens:
        print(f"GT canonical row length: n={len(lens)} mean={np.mean(lens):.1f} med={np.median(lens):.0f} "
              f"p10={np.percentile(lens,10):.0f} p90={np.percentile(lens,90):.0f} min={min(lens)} max={max(lens)}")
    if cond_mismatch_ids:
        print("examples of cond mismatch:")
        for pid, c, p in cond_mismatch_ids:
            print(f"   {pid}: canon={c!r} official={p!r}")
    fails = [(k, v["failed"]) for k, v in rows_json.items() if "failed" in v]
    if fails:
        print(f"failed examples ({len(fails)}):", fails[:5])
    print(f"output: {out_dir}/  {os.path.join(HERE,'data',args.ds,f'gt_rows{args.out_suffix}.json')}")


if __name__ == "__main__":
    main()
