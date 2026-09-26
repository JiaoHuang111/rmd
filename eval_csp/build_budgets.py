"""
build_budgets.py — computes the generation length budget for each prompt.

Rationale: fixed-length diffusion cannot stop on its own; a candidate much longer than the
true structure carries junk tokens, while a shorter one truncates the atom rows. Anchor on
"the original token-length distribution of the training rows with the same reduced
composition":
  budget = median_len(same-comp training rows) - cond_len + slack
Only the composition is used for conditioning, so using the composition's own training length
distribution stays closest to the behaviour of the CrystaLLM autoregressive model, which
decides its length from its own prior. Prompts whose composition is absent from the training
data (compositions unique to a dataset's test split) fall back to the dataset-level median
length and are recorded.

Original row lengths are read directly from the number of decoded rows in
tokens <ds>/{train,val}.bin -- independent of the representation.
"""
import argparse
import array
import collections
import importlib.util
import json
import os
import re
import statistics

from pymatgen.core import Composition

# Load the tokenizer straight from the file (byte-for-byte identical to
# CrystaLLM-main/crystallm/_tokenizer.py) to avoid pulling in the deep dependencies of the
# byprot package.
_spec = importlib.util.spec_from_file_location(
    "cif_tokenizer", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "../src/byprot/crystallm/_tokenizer.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
CIFTokenizer = _mod.CIFTokenizer
_TOKENIZER = CIFTokenizer()


def comp_of_row_text(text):
    """data_ header of a tokens row (full-cell style) -> reduced formula."""
    m = re.match(r"data_([^\n]+)\n", text)
    if not m:
        return None
    try:
        return Composition(m.group(1)).reduced_formula
    except Exception:
        return None


def data_id_of(itos):
    """Token id of the row-leading 'data_' (identical for all 4 datasets in meta.pkl, =124)."""
    if isinstance(itos, list):
        return itos.index("data_")
    return [i for i, t in itos.items() if t == "data_"][0]


def derive_row_starts(tokens_dir, split, itos):
    """Position of the data_ token = row start (see row_starts_of_bin's docstring for the
    justification; the token does not occur again inside a row)."""
    a = array.array("H")
    a.frombytes(open(os.path.join(tokens_dir, f"{split}.bin"), "rb").read())
    did = data_id_of(itos)
    return [i for i, t in enumerate(a) if t == did]


def row_starts_of_bin(tokens_dir, ds_name, split, itos):
    """Row boundaries: prefer the official starts_<ds>_<split>.pkl; if missing, derive them
    from the position of the row-leading token.

    Justification (verified): every token row starts with 'data_' and that token does not
    occur again within the row (for mp_20 train+val the official starts equal the data_
    positions one by one, 9047/9047; 0 occurrences inside rows).
    For val splits without starts, the same rule is only used after the train split has
    passed the official-vs-derived equality check in main()."""
    pkl = os.path.join(tokens_dir, f"starts_{ds_name}_{split}.pkl")
    if os.path.exists(pkl):
        return sorted(pickle_load(pkl)), "official"
    return derive_row_starts(tokens_dir, split, itos), "derived(data_)"


def load_token_lens(tokens_dir, split):
    meta = pickle_load(os.path.join(tokens_dir, "meta.pkl"))
    itos = meta["itos"]
    ds_name = os.path.basename(tokens_dir).replace("tokens_", "")
    starts, basis = row_starts_of_bin(tokens_dir, ds_name, split, itos)
    a = array.array("H")
    a.frombytes(open(os.path.join(tokens_dir, f"{split}.bin"), "rb").read())
    arr = a.tolist()

    def dec(ids):
        return "".join(itos[i] for i in ids)

    out = []
    for i in range(len(starts) - 1):
        out.append((comp_of_row_text(dec(arr[starts[i]:starts[i + 1]])),
                    starts[i + 1] - starts[i]))
    out.append((comp_of_row_text(dec(arr[starts[-1]:])), len(arr) - starts[-1]))
    return out, basis


def pickle_load(p):
    import pickle
    return pickle.load(open(p, "rb"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--tokens-dir", required=True,
                    help="e.g. data-bin/crystalmols/tokens_<ds>")
    ap.add_argument("--prompts-dir", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--slack", type=int, default=8)
    ap.add_argument("--min-budget", type=int, default=50)
    ap.add_argument("--max-total", type=int, default=1005, help="upper bound on cond+budget (ESM pos emb 1026 leaves margin)")
    args = ap.parse_args()

    train_rows, train_basis = load_token_lens(args.tokens_dir, "train")
    # Sanity check: the derived train boundaries must be exactly equal to the official
    # starts before the same rule is applied to a val split lacking starts
    meta = pickle_load(os.path.join(args.tokens_dir, "meta.pkl"))
    itos = meta["itos"]
    derived_train = derive_row_starts(args.tokens_dir, "train", itos)
    official_train = sorted(pickle_load(
        os.path.join(args.tokens_dir,
                     f"starts_{os.path.basename(args.tokens_dir).replace('tokens_', '')}_train.pkl")))
    assert derived_train == official_train, \
        f"{args.ds}: derived train starts != official starts!"
    val_rows, val_basis = load_token_lens(args.tokens_dir, "val")
    rows = train_rows + val_rows
    by_comp = collections.defaultdict(list)
    n_unk = 0
    for comp, ln in rows:
        if comp is None:
            n_unk += 1
            continue
        by_comp[comp].append(ln)
    all_lens = [ln for _, ln in rows if ln is not None]
    global_median = statistics.median(all_lens)
    print(f"{args.ds}: token rows={len(rows)} (unparsed {n_unk}), "
          f"comps={len(by_comp)}, len median={global_median} "
          f"(min/max {min(all_lens)}/{max(all_lens)}); "
          f"train basis={train_basis}, val basis={val_basis}")

    cond_re = re.compile(r"^data_([A-Za-z0-9]+)$")
    budgets, meta = {}, {}
    n_comp_miss = 0
    n_clamped = 0
    for fn in sorted(os.listdir(args.prompts_dir)):
        if not fn.endswith(".txt"):
            continue
        pid = fn[:-4]
        text = open(os.path.join(args.prompts_dir, fn), encoding="utf-8").read().strip()
        m = cond_re.match(text)
        assert m, f"{pid}: bad prompt {text!r}"
        formula = m.group(1)
        try:
            comp = Composition(formula).reduced_formula
        except Exception:
            comp = None
        # cond length: same tokenizer rule as for the training row prefix (tokenize the
        # prompt text directly)
        cond_toks = _TOKENIZER.tokenize_cif(text + "\n")
        assert all(t in _TOKENIZER.token_to_id for t in cond_toks), f"{pid}: unk in cond"
        cond_len = len(cond_toks)
        lens = by_comp.get(comp)
        if not lens:
            n_comp_miss += 1
            base = global_median
            meta[pid] = {"comp": comp, "n_train_rows": 0, "comp_miss": True}
        else:
            base = statistics.median(lens)
            meta[pid] = {"comp": comp, "n_train_rows": len(lens), "comp_miss": False}
        budget = int(round(base)) - cond_len + args.slack
        budget = max(args.min_budget, budget)
        total = cond_len + budget
        if total > args.max_total:
            budget = args.max_total - cond_len
            n_clamped += 1
        budgets[pid] = budget
        meta[pid].update({"formula": formula, "cond_len": cond_len,
                          "train_median_len": int(round(base)), "budget": budget})

    print(f"prompts={len(budgets)} comp-miss(→global median)={n_comp_miss} "
          f"clamped={n_clamped}")
    bl = list(budgets.values())
    print(f"budget: median={statistics.median(bl)} min={min(bl)} max={max(bl)}")
    with open(args.out_json, "w") as f:
        json.dump(budgets, f, indent=0)
    with open(args.out_json.replace("budgets", "budget_meta"), "w") as f:
        json.dump({"global_median_len": global_median,
                   "length_basis": {"train": train_basis, "val": val_basis},
                   "prompts": meta}, f, indent=1)
    print(f"saved {args.out_json}")


if __name__ == "__main__":
    main()
