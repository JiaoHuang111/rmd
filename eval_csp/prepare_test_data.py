"""
prepare_test_data.py — stage 1: extract the per-prompt evaluation data from the benchmark
test.csv.

For each dataset <ds>:
  1. Read the `cif` column (the original GT CIF) of CrystaLLM's read-only
     resources/benchmarks/<ds>/test.csv.
  2. Write eval_csp/data/<ds>/orig/<id>.cif keyed by material_id (text identical to the csv).
  3. Produce eval_csp/data/<ds>/test_input.pkl.gz = [(id, cif)] so that CrystaLLM's own
     bin/preprocess.py (run in read-only mode, its repository unmodified) can rebuild the
     prep version of the CIFs.

An evaluation unit = an id in the CrystaLLM prompt tars (prompts/<ds>/*.txt matches the
test.csv material_id; the mpts_52 csv has one extra id, mp-1056831, whose prompt is absent
from CrystaLLM's own tars, so it is skipped as well, following the CrystaLLM convention and
keeping this set fully isomorphic to the official prompt set).
"""
import csv
import gzip
import json
import os
import pickle
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from _paths import crystallm_csvroot  # noqa: E402

# The read-only CrystaLLM checkout that supplies the benchmark CSVs.
CSVROOT = crystallm_csvroot()
PROMPTS = os.path.join(ROOT, "prompts")
DATA = os.path.join(ROOT, "data")

DATASETS = ["mp_20", "carbon_24", "mpts_52", "perov_5"]


def main():
    for ds in DATASETS:
        rows = list(csv.DictReader(open(f"{CSVROOT}/{ds}/test.csv")))
        prompt_ids = sorted(
            f[:-4] for f in os.listdir(os.path.join(PROMPTS, ds)) if f.endswith(".txt")
        )
        prompt_set = set(prompt_ids)
        assert len(prompt_ids) == len(prompt_set)

        by_id = {}
        for r in rows:
            by_id[r["material_id"]] = r["cif"]
        assert prompt_set <= set(by_id), (
            f"{ds}: {len(prompt_set - set(by_id))} prompt ids missing from csv")
        extra = set(by_id) - prompt_set
        if extra:
            print(f"{ds}: skipping csv ids without prompts: {sorted(extra)}")

        orig_dir = os.path.join(DATA, ds, "orig")
        os.makedirs(orig_dir, exist_ok=True)
        pairs = []
        for i, pid in enumerate(prompt_ids):
            text = by_id[pid].strip() + "\n"
            with open(os.path.join(orig_dir, f"{pid}.cif"), "w") as f:
                f.write(text)
            pairs.append((pid, text))
        out = os.path.join(DATA, ds, "test_input.pkl.gz")
        with gzip.open(out, "wb") as f:
            pickle.dump(pairs, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"{ds}: {len(pairs)} prompts | orig cifs -> {orig_dir} | input -> {out}")


if __name__ == "__main__":
    main()
