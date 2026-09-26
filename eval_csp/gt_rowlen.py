"""GT canonical row length: reproduces the CrystaLLM corpus construction chain
raw CIF -> [replace_data_formula_with_nonreduced_formula, semisymmetrize_cif,
           add_atomic_props_block, round_numbers(4)]
        -> [strip each line, drop empty / '#' / pymatgen lines, append "\n", join]  (tokenize_cifs.py::preprocess)
        -> CIFTokenizer.tokenize_cif -> token string
Row length = token count (consistent with the tokens_mp_20 row definition: no eos, ending
with the two tokens '\n','\n').
"""
import importlib.util, json, os, re

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
U = load("cifutils", os.path.join(ROOT, "src/byprot/crystallm/_utils.py"))
TK = load("ciftok", os.path.join(ROOT, "src/byprot/crystallm/_tokenizer.py"))
tok = TK.CIFTokenizer()

def canonicalize(cif_str):
    cif_str = U.replace_data_formula_with_nonreduced_formula(cif_str)
    cif_str = U.semisymmetrize_cif(cif_str)
    cif_str = U.add_atomic_props_block(cif_str, oxi=False)
    return U.round_numbers(cif_str, decimal_places=4)

def line_clean(cif_str):
    lines = []
    for line in cif_str.split("\n"):
        line = line.strip()
        if len(line) > 0 and not line.startswith("#") and "pymatgen" not in line:
            lines.append(line)
    lines.append("\n")
    return "\n".join(lines)

budgets = json.load(open(os.path.join(HERE, "data/mp_20/budgets.json")))
meta = json.load(open(os.path.join(HERE, "data/mp_20/budget_meta.json")))["prompts"]
ids = ["mp-10009", "mp-1001", "mp-1001012", "mp-1001034", "mp-10014"]
out = {}
print(f"{'id':<12} {'cond':>4} {'budget':>6} {'GT_row':>6} {'GT_mask':>7} {'Δ(mask-budget)':>15} {'train_med':>9} {'miss':>5}")
for pid in ids:
    raw = open(os.path.join(HERE, f"data/mp_20/orig/{pid}.cif")).read()
    can = canonicalize(raw); text = line_clean(can)
    toks = tok.tokenize_cif(text)
    unks = sum(1 for t in toks if t == "<unk>")
    m = meta[pid]
    gt_mask = len(toks) - m["cond_len"]
    print(f"{pid:<12} {m['cond_len']:>4} {m['budget']:>6} {len(toks):>6} {gt_mask:>7} "
          f"{gt_mask - m['budget']:>15} {m['train_median_len']:>9} {int(m['comp_miss']):>5}"
          + (f"  unk={unks}" if unks else ""))
    out[pid] = {"gt_row_len": len(toks), "gt_mask_len": gt_mask, "budget": m["budget"],
                "cond_len": m["cond_len"], "unk": unks, "head": can.split("\n")[0]}
json.dump(out, open(os.path.join(HERE, "data/mp_20/gt_rowlen.json"), "w"), indent=1)
print("\nheaders:", {k: v["head"] for k, v in out.items()})
