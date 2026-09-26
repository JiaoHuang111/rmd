"""1) Cross-check: the data_ line of canonicalize(raw GT) vs the official prompt text (the
      official prompts are extracted from the official preprocessed CIFs)
   2) Length distribution of the corpus rows sharing the same cond (data_ line tokens equal
      one by one), and the length distribution for the same reduced comp"""
import importlib.util, json, os, pickle, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from _paths import tokens_dir  # noqa: E402
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); return m
U = load("cifutils", os.path.join(ROOT, "src/byprot/crystallm/_utils.py"))
TK = load("ciftok", os.path.join(ROOT, "src/byprot/crystallm/_tokenizer.py"))
tok = TK.CIFTokenizer()
TOKDIR = tokens_dir("mp_20")
ids = ["mp-10009", "mp-1001", "mp-1001012", "mp-1001034", "mp-10014"]

def line_clean(s):
    lines = [l.strip() for l in s.split("\n") if len(l.strip()) > 0 and not l.strip().startswith("#") and "pymatgen" not in l]
    lines.append("\n"); return "\n".join(lines)

def canon(raw):
    s = U.replace_data_formula_with_nonreduced_formula(raw)
    s = U.semisymmetrize_cif(s)
    s = U.add_atomic_props_block(s, oxi=False)
    return U.round_numbers(s, decimal_places=4)

# ---- corpus rows ----
meta = pickle.load(open(os.path.join(TOKDIR, "meta.pkl"), "rb"))
stoi = meta["stoi"]; data_id = stoi["data_"]
tr = np.fromfile(os.path.join(TOKDIR, "train.bin"), dtype=np.uint16)
va = np.fromfile(os.path.join(TOKDIR, "val.bin"), dtype=np.uint16)
starts = {}
for split, arr in (("train", tr), ("val", va)):
    sp = os.path.join(TOKDIR, f"starts_mp_20_{split}.pkl")
    starts[split] = np.array(pickle.load(open(sp, "rb")), dtype=np.int64)

def rows_of(arr, st):
    st = np.sort(st); ends = np.append(st[1:], len(arr))
    return [(int(s), int(e)) for s, e in zip(st, ends)]

ROWS = {sp: rows_of(tr if sp == "train" else va, starts[sp]) for sp in ("train", "val")}
ito = {int(k): v for k, v in meta["itos"].items()}
def row_tokens(sp, s, e): return [ito[int(t)] for t in (tr if sp == "train" else va)[s:e]]

# build the cond index: leading data_ line token string -> list of row lengths
from collections import defaultdict
cond2lens = defaultdict(list)
for sp in ("train", "val"):
    for s, e in ROWS[sp]:
        toks = row_tokens(sp, s, e)
        # data_ line = up to and including the first \n (a further '\n' too?) -- from the
        # leading token through the first '\n'
        try: nl = toks.index("\n")
        except ValueError: nl = len(toks)
        cond2lens[tuple(toks[:nl + 1])].append(e - s)

def reduced_key(toks):
    """data_ line tokens -> reduced composition key (elements + smallest stoichiometric ratio)"""
    from math import gcd
    els, nums, cur = [], [], None
    for t in toks[1:]:
        if t == "\n": break
        if t.isdigit(): nums[-1] = nums[-1] * 10 + int(t)
        else: els.append(t); nums.append(0)
    ns = [n if n > 0 else 1 for n in nums]
    if not ns: return None
    g = ns[0]
    for n in ns[1:]: g = gcd(g, n)
    return tuple(x for pair in zip(els, [n // g for n in ns]) for x in pair)

red2lens = defaultdict(list)
for sp in ("train", "val"):
    for s, e in ROWS[sp]:
        toks = row_tokens(sp, s, e)
        try: nl = toks.index("\n")
        except ValueError: continue
        k = reduced_key(toks[:nl + 1])
        if k: red2lens[k].append(e - s)

gt = json.load(open(os.path.join(HERE, "data/mp_20/gt_rowlen.json")))
print("=== 1) canonical data_ line vs official prompt ===")
out = {}
for pid in ids:
    raw = open(os.path.join(HERE, f"data/mp_20/orig/{pid}.cif")).read()
    c = canon(raw); text = line_clean(c)
    toks = tok.tokenize_cif(text)
    try: nl = toks.index("\n")
    except ValueError: nl = len(toks)
    canon_cond = "".join(toks[:nl + 1])
    official = open(os.path.join(HERE, f"prompts/mp_20/{pid}.txt")).read()
    ok = canon_cond == official
    ctoks = tuple(toks[:nl + 1])
    key = reduced_key(toks[:nl + 1])
    lens_same = sorted(cond2lens.get(ctoks, []))
    lens_red = sorted(red2lens.get(key, []))
    print(f"{pid}: data_ line==official prompt: {ok}  ({canon_cond.strip()!r} vs {official.strip()!r})" if not ok else
          f"{pid}: data_ line==official prompt: True  ({official.strip()!r})")
    print(f"    GT_row={gt[pid]['gt_row_len']:>4}  budget={gt[pid]['budget']:>4}  "
          f"same-cond rows: n={len(lens_same)} med={np.median(lens_same) if lens_same else None} "
          f"p10={np.percentile(lens_same,10) if lens_same else None} p90={np.percentile(lens_same,90) if lens_same else None} "
          f"min={lens_same[0] if lens_same else None} max={lens_same[-1] if lens_same else None}")
    print(f"    same-reduced-comp rows: n={len(lens_red)} med={np.median(lens_red) if lens_red else None} "
          f"p10={np.percentile(lens_red,10) if lens_red else None} p90={np.percentile(lens_red,90) if lens_red else None} "
          f"min={lens_red[0] if lens_red else None} max={lens_red[-1] if lens_red else None}")
    out[pid] = {"canon_cond": canon_cond, "official": official, "match": bool(ok),
                "same_cond_n": len(lens_same), "same_cond_lens": lens_same[:200],
                "same_red_n": len(lens_red)}
json.dump(out, open(os.path.join(HERE, "data/mp_20/cond_rowstats.json"), "w"), indent=1)
