"""
postprocess_candidates.py — verbatim port of the official bin/postprocess.py (implemented
only inside this directory; the CrystaLLM repository is not modified):

  postprocess(cif) =
      space_group_symbol = extract_space_group_symbol(cif)
      if space_group_symbol is not None and space_group_symbol != "P 1":
          cif = replace_symmetry_operators(cif, space_group_symbol)   # replace the symmetry ops with the declared space group
      cif = remove_atom_props_block(cif)                              # drop the _atom_type_* property block

Purpose: DMLM-generated candidates are "corpus style" (the symmetry operations list only
'x, y, z' while the sites keep their multiplicity), so pymatgen only expands the operations
that are listed when parsing -> the site count is underestimated. The official pipeline
requires postprocess to run first (the user's existing pipeline: run_postprocess.py ->
post_cifs_*.tar.gz -> evaluate_cifs.py); this script converts the candidates to the official
evaluation definition.

Usage: python postprocess_candidates.py <in_dir> <out_dir> [--limit N]
"""
import argparse
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "cifutils", os.path.join(HERE, "..", "src", "byprot", "crystallm", "_utils.py"))
U = importlib.util.module_from_spec(spec)
spec.loader.exec_module(U)


def postprocess(cif):
    sg = U.extract_space_group_symbol(cif)
    if sg is not None and sg != "P 1":
        cif = U.replace_symmetry_operators(cif, sg)
    cif = U.remove_atom_props_block(cif)
    return cif


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(args.in_dir) if f.endswith(".cif"))
    if args.limit:
        files = files[:args.limit]
    n_ok = n_warn = 0
    for i, f in enumerate(files):
        cif = open(os.path.join(args.in_dir, f)).read()
        try:
            out = postprocess(cif)
            n_ok += 1
        except Exception as e:
            out = f"# WARNING: postprocess failed: {e}\n" + cif
            n_warn += 1
        with open(os.path.join(args.out_dir, f), "w") as fh:
            fh.write(out)
        if (i + 1) % 200 == 0:
            print(f"  ... {i+1}/{len(files)}", flush=True)
    print(f"✓ postprocessed {n_ok} ok, {n_warn} warnings -> {args.out_dir}")


if __name__ == "__main__":
    main()
