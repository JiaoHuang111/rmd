"""Validation: val.csv -> preprocess_test.augment_cif -> line_clean -> tokenize, compared
against the 9047 rows of tokens_mp_20/val.bin in terms of text set / length distribution
(a credibility check of the GT row-length protocol)."""
import csv, pickle, numpy as np, importlib.util, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("pt", "preprocess_test.py"); pt = importlib.util.module_from_spec(spec); spec.loader.exec_module(pt)
spec2 = importlib.util.spec_from_file_location("tk", "../src/byprot/crystallm/_tokenizer.py")
from _paths import crystallm_csv, tokens_dir; TK = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(TK)
tok = TK.CIFTokenizer()
def line_clean(s):
    ls = [l.strip() for l in s.split("\n") if len(l.strip()) > 0 and not l.strip().startswith("#") and "pymatgen" not in l]
    ls.append("\n"); return "\n".join(ls)
rows = list(csv.DictReader(open(crystallm_csv("mp_20", "val"))))
mine = {}
for r in rows:
    try: can = pt.augment_cif(r["material_id"], r["cif"])
    except Exception: continue
    toks = tok.tokenize_cif(line_clean(can))
    mine[r["material_id"]] = (len(toks), "".join(toks))
TOKDIR = tokens_dir("mp_20")
meta = pickle.load(open(f"{TOKDIR}/meta.pkl", "rb")); itos = {int(k): v for k, v in meta["itos"].items()}
arr = np.fromfile(f"{TOKDIR}/val.bin", dtype=np.uint16)
st = np.sort(np.array(pickle.load(open(f"{TOKDIR}/starts_mp_20_val.pkl", "rb")), dtype=np.int64))
corpus = [("".join(itos[int(t)] for t in arr[s:e]), e - s) for s, e in zip(st, np.append(st[1:], len(arr)))]
lm = [v[0] for v in mine.values()]; lc = [c[1] for c in corpus]
print(f"len  mine: n={len(lm)} mean={np.mean(lm):.1f} med={np.median(lm)} p10={np.percentile(lm,10):.0f} p90={np.percentile(lm,90):.0f}")
print(f"len  corp: n={len(lc)} mean={np.mean(lc):.1f} med={np.median(lc)} p10={np.percentile(lc,10):.0f} p90={np.percentile(lc,90):.0f}")
setm = {v[1] for v in mine.values()}; setc = {c[0] for c in corpus}
print(f"text set overlap: {len(setm & setc)} / {len(setc)}   (mine={len(setm)} corp={len(setc)})")
if setc - setm:
    ex = sorted(setc - setm, key=len)[0]
    nm = min(mine.values(), key=lambda m: abs(m[0] - len(tok.tokenize_cif(ex))))
    print(f"=== corpus-only example (len={len(tok.tokenize_cif(ex))}) ==="); print(ex[:500])
    print(f"=== our text with the closest length (len={nm[0]}) ==="); print(nm[1][:500])
