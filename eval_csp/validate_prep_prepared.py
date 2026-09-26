"""Validation: val.csv --(official standardization CifWriter symprec=0.1)--> preprocess -->
tokenize, compared against the 9047 rows of corpus tokens_mp_20/val.bin in terms of text
set / length."""
import csv, gzip, pickle, importlib.util, os, sys, numpy as np, multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import crystallm_csv, tokens_dir
pt = importlib.util.spec_from_file_location("pt", "preprocess_test.py")
pt = importlib.util.module_from_spec(pt)
spec2 = importlib.util.spec_from_file_location("tk", "../src/byprot/crystallm/_tokenizer.py")
TK = importlib.util.module_from_spec(spec2)
NK = 2
def _init():
    global _pt, _tk, _tok
    global _spec
    s = importlib.util.spec_from_file_location("pt2", "preprocess_test.py"); _pt = importlib.util.module_from_spec(s); s.loader.exec_module(_pt)
    s2 = importlib.util.spec_from_file_location("tk2", "../src/byprot/crystallm/_tokenizer.py"); _tk = importlib.util.module_from_spec(s2); s2.loader.exec_module(_tk)
    _tok = _tk.CIFTokenizer()
def line_clean(s):
    ls=[l.strip() for l in s.split("\n") if len(l.strip())>0 and not l.strip().startswith("#") and "pymatgen" not in l]
    ls.append("\n"); return "\n".join(ls)
def work(item):
    from pymatgen.io.cif import CifWriter, Structure
    pid, cif = item
    try:
        prepared = str(CifWriter(struct=Structure.from_str(cif, fmt="cif"), symprec=0.1))
        can = _pt.augment_cif(pid, prepared)
        return pid, "".join(_tok.tokenize_cif(line_clean(can)))
    except Exception as e:
        return pid, None
if __name__ == "__main__":
    rows = list(csv.DictReader(open(crystallm_csv("mp_20", "val"))))
    items = [(r["material_id"], r["cif"]) for r in rows]
    with mp.Pool(NK, initializer=_init) as pool:
        mine = dict(pool.imap_unordered(work, items, chunksize=32))
    TOKDIR = tokens_dir("mp_20")
    meta=pickle.load(open(f"{TOKDIR}/meta.pkl","rb")); itos={int(k):v for k,v in meta["itos"].items()}
    arr=np.fromfile(f"{TOKDIR}/val.bin",dtype=np.uint16)
    st=np.sort(np.array(pickle.load(open(f"{TOKDIR}/starts_mp_20_val.pkl","rb")),dtype=np.int64))
    corpus=["".join(itos[int(t)] for t in arr[s:e]) for s,e in zip(st,np.append(st[1:],len(arr)))]
    ok=[v for v in mine.values() if v]
    lm=[len(v) for v in ok]; lc=[len(c) for c in corpus]
    print(f"prepared rows: n={len(ok)}/{len(items)}  len mean={np.mean(lm):.1f} med={np.median(lm):.0f} p10={np.percentile(lm,10):.0f} p90={np.percentile(lm,90):.0f}")
    print(f"corpus  val: n={len(corpus)}    len mean={np.mean(lc):.1f} med={np.median(lc):.0f} p10={np.percentile(lc,10):.0f} p90={np.percentile(lc,90):.0f}")
    print(f"text set overlap: {len(set(ok) & set(corpus))} / {len(set(corpus))}")
    miss = set(corpus) - set(ok)
    if miss:
        ex = sorted(miss, key=len)[0]
        print(f"=== corpus-only example (len={len(ex)}) ==="); print(ex[:400])
