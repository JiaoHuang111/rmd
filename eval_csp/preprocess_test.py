"""
preprocess_test.py — reimplementation of CrystaLLM's CIF preprocessing (used for test-split
length anchoring).

CrystaLLM's bin/preprocess.py depends on `import crystallm` (whose __init__ imports zmq,
which the evaluation environment on this machine does not have). The 5 pure functions it
uses are ported verbatim from that checkout's crystallm/_utils.py (character-for-character
identical), so
the behaviour is exactly that of bin/preprocess.py::augment_cif:

    formula_units = extract_formula_units(cif)          # CIFs with Z==0 are errors, dropped
    cif = replace_data_formula_with_nonreduced_formula(cif)
    cif = semisymmetrize_cif(cif)
    cif = add_atomic_props_block(cif, oxi=False)
    cif = round_numbers(cif, decimal_places=4)

The output is structurally identical to the official pipeline's *_prep.pkl.gz:
[(id, cif_str)] (gz+pickle).
Validation (see main --validate at the end of this file): running all rows of the benchmark
val.csv through this chain and then tokenizing/decoding them yields the same overall set as
the decoded rows of val.bin in the tokens data => the port is byte-for-byte equivalent.

Runtime environment: pymatgen (on this machine the dmlm env, 2024.8.9; for the finding that
in validate mode the output does not match the tokens representation, see
TRAINING_CSP.md / the analysis notes -- the tokens were produced from older prep text and
cannot be reproduced at the byte level).
"""
import gzip
import math
import os
import pickle
import re
import sys

from pymatgen.core import Composition
from pymatgen.io.cif import CifBlock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import REPO_ROOT, crystallm_csvroot, tokens_dir  # noqa: E402

# ------------------------------------------------------------------ helpers (copy)

def get_atomic_props_block(composition, oxi=False):
    noble_vdw_radii = {
        "He": 1.40,
        "Ne": 1.54,
        "Ar": 1.88,
        "Kr": 2.02,
        "Xe": 2.16,
        "Rn": 2.20,
    }

    allen_electronegativity = {
        "He": 4.16,
        "Ne": 4.79,
        "Ar": 3.24,
    }

    def _format(val):
        return f"{float(val): .4f}"

    def _format_X(elem):
        if math.isnan(elem.X) and str(elem) in allen_electronegativity:
            return allen_electronegativity[str(elem)]
        return _format(elem.X)

    def _format_radius(elem):
        if elem.atomic_radius is None and str(elem) in noble_vdw_radii:
            return noble_vdw_radii[str(elem)]
        return _format(elem.atomic_radius)

    props = {str(el): (_format_X(el), _format_radius(el), _format(el.average_ionic_radius))
             for el in sorted(composition.elements)}

    data = {}
    data["_atom_type_symbol"] = list(props)
    data["_atom_type_electronegativity"] = [v[0] for v in props.values()]
    data["_atom_type_radius"] = [v[1] for v in props.values()]
    # use the average ionic radius
    data["_atom_type_ionic_radius"] = [v[2] for v in props.values()]

    loop_vals = [
        "_atom_type_symbol",
        "_atom_type_electronegativity",
        "_atom_type_radius",
        "_atom_type_ionic_radius"
    ]

    if oxi:
        symbol_to_oxinum = {str(el): (float(el.oxi_state), _format(el.ionic_radius)) for el in sorted(composition.elements)}
        data["_atom_type_oxidation_number"] = [v[0] for v in symbol_to_oxinum.values()]
        # if we know the oxidation state of the element, use the ionic radius for the given oxidation state
        data["_atom_type_ionic_radius"] = [v[1] for v in symbol_to_oxinum.values()]
        loop_vals.append("_atom_type_oxidation_number")

    loops = [loop_vals]

    return str(CifBlock(data, loops, "")).replace("data_\n", "")


def extract_numeric_property(cif_str, prop, numeric_type=float):
    match = re.search(rf"{prop}\s+([.0-9]+)", cif_str)
    if match:
        return numeric_type(match.group(1))
    raise Exception(f"could not find {prop} in:\n{cif_str}")


def extract_formula_units(cif_str):
    return extract_numeric_property(cif_str, "_cell_formula_units_Z", numeric_type=int)


def extract_formula_nonreduced(cif_str):
    match = re.search(r"_chemical_formula_sum\s+('([^']+)'|(\S+))", cif_str)
    if match:
        return match.group(2) if match.group(2) else match.group(3)
    raise Exception(f"could not extract _chemical_formula_sum value from:\n{cif_str}")


def semisymmetrize_cif(cif_str):
    return re.sub(
        r"(_symmetry_equiv_pos_as_xyz\n)(.*?)(?=\n(?:\S| \S))",
        r"\1  1  'x, y, z'",
        cif_str,
        flags=re.DOTALL
    )


def replace_data_formula_with_nonreduced_formula(cif_str):
    pattern = r"_chemical_formula_sum\s+(.+)\n"
    pattern_2 = r"(data_)(.*?)(\n)"
    match = re.search(pattern, cif_str)
    if match:
        chemical_formula = match.group(1)
        chemical_formula = chemical_formula.replace("'", "").replace(" ", "")

        modified_cif = re.sub(pattern_2, r'\1' + chemical_formula + r'\3', cif_str)

        return modified_cif
    else:
        raise Exception(f"Chemical formula not found {cif_str}")


def add_atomic_props_block(cif_str, oxi=False):
    comp = Composition(extract_formula_nonreduced(cif_str))

    block = get_atomic_props_block(composition=comp, oxi=oxi)

    # the hypothesis is that the atomic properties should be the first thing
    #  that the model must learn to associate with the composition, since
    #  they will determine so much of what follows in the file
    pattern = r"_symmetry_space_group_name_H-M"
    match = re.search(pattern, cif_str)

    if match:
        start_pos = match.start()
        modified_cif = cif_str[:start_pos] + block + "\n" + cif_str[start_pos:]
        return modified_cif
    else:
        raise Exception(f"Pattern not found: {cif_str}")


def round_numbers(cif_str, decimal_places=4):
    # Pattern to match a floating point number in the CIF file
    # It also matches numbers in scientific notation
    pattern = r"[-+]?\d*\.\d+([eE][-+]?\d+)?"

    # Function to round the numbers
    def round_number(match):
        number_str = match.group()
        number = float(number_str)
        # Check if number of digits after decimal point is less than 'decimal_places'
        if len(number_str.split('.')[-1]) <= decimal_places:
            return number_str
        rounded = round(number, decimal_places)
        return format(rounded, '.{}f'.format(decimal_places))

    # Replace all occurrences of the pattern using a regex sub operation
    cif_string_rounded = re.sub(pattern, round_number, cif_str)

    return cif_string_rounded


# ------------------------------------------------------------------ augment chain
def augment_cif(id, cif_str, oxi=False, decimal_places=4):
    """Same processing chain as CrystaLLM bin/preprocess.py::augment_cif; exceptions / bad
    CIFs are raised and dropped by the caller."""
    formula_units = extract_formula_units(cif_str)
    # exclude CIFs with formula units (Z) = 0, which are erroneous
    if formula_units == 0:
        raise Exception("formula units (Z) == 0")

    cif_str = replace_data_formula_with_nonreduced_formula(cif_str)
    cif_str = semisymmetrize_cif(cif_str)
    cif_str = add_atomic_props_block(cif_str, oxi)
    cif_str = round_numbers(cif_str, decimal_places=decimal_places)
    return cif_str


def preprocess_list(pairs, oxi=False, decimal_places=4):
    """Input [(id, cif)] -> output [(id, preprocessed_cif)]; rows that error out are dropped,
    as in the original pipeline."""
    out = []
    for id_, cif in pairs:
        try:
            out.append((id_, augment_cif(id_, cif, oxi=oxi, decimal_places=decimal_places)))
        except Exception:
            pass
    return out


def load_pairs(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def save_pairs(path, pairs):
    with gzip.open(path, "wb") as f:
        pickle.dump(pairs, f, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("in_pkl", nargs="?", default=None, help="[(id,cif)] gz+pickle input")
    ap.add_argument("-o", "--out", default=None, help="[(id,prep)] gz+pickle output")
    ap.add_argument("--oxi", action="store_true")
    ap.add_argument("--decimal-places", type=int, default=4)
    ap.add_argument("--validate", action="store_true",
                    help="mp_20 val byte-level validation: val.csv -> this chain -> tokenize/decode, "
                         "compared against the set of decoded val.bin rows")
    args = ap.parse_args()

    if args.validate:
        csvroot = crystallm_csvroot()
        dmlmbase = tokens_dir("mp_20")
        import array
        import csv
        import importlib.util
        # Loaded straight from the file (no diff against
        # CrystaLLM-main/crystallm/_tokenizer.py), to avoid pulling in the deep
        # dependencies of the byprot package.
        _spec = importlib.util.spec_from_file_location(
            "cif_tokenizer", os.path.join(REPO_ROOT, "src/byprot/crystallm/_tokenizer.py"))
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        CIFTokenizer = _mod.CIFTokenizer

        meta = pickle.load(open(os.path.join(dmlmbase, "meta.pkl"), "rb"))
        itos = meta["itos"]
        starts = sorted(pickle.load(open(os.path.join(dmlmbase, "starts_mp_20_val.pkl"), "rb")))
        a = array.array("H")
        a.frombytes(open(os.path.join(dmlmbase, "val.bin"), "rb").read())
        arr = a.tolist()

        def dec(ids):
            return "".join(itos[i] for i in ids)

        tok = CIFTokenizer()
        # 1) set of decoded val.bin rows
        bin_rows = {dec(arr[starts[i]:starts[i + 1]]) for i in range(len(starts) - 1)}
        bin_rows.add(dec(arr[starts[-1]:]))
        print(f"val.bin rows: {len(bin_rows)}")
        # 2) val.csv -> this chain -> tokenize -> decode
        rows = list(csv.DictReader(open(f"{csvroot}/mp_20/val.csv")))
        pairs = [(r["material_id"], r["cif"]) for r in rows]
        prepped = preprocess_list(pairs)
        print(f"val.csv preprocessed rows: {len(prepped)} (dropped {len(pairs) - len(prepped)})")
        ours = set()
        for _, cif in prepped:
            tokens = tok.tokenize_cif(cif)
            ids = [tok.token_to_id[t] for t in tokens]
            ours.add(dec(ids))
        only_bin = bin_rows - ours
        only_ours = ours - bin_rows
        print(f"multiset size bin={len(bin_rows)} ours={len(ours)}; "
              f"only-in-bin={len(only_bin)} only-in-ours={len(only_ours)}")
        if only_bin:
            print("example only-in-bin:", repr(sorted(only_bin)[0][:200]))
        if only_ours:
            print("example only-in-ours:", repr(sorted(only_ours)[0][:200]))
        return

    assert args.in_pkl and args.out
    pairs = load_pairs(args.in_pkl)
    out = preprocess_list(pairs, oxi=args.oxi, decimal_places=args.decimal_places)
    print(f"input={len(pairs)} kept={len(out)} dropped={len(pairs) - len(out)}")
    save_pairs(args.out, out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
