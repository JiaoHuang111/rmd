"""
csp_metrics.py — CSP evaluation metrics (Match Rate / RMSE), ported from the CSP branch
of CrystaLLM's bin/benchmark_metrics.py and kept exactly consistent:

  * StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)  -- comment "defaults taken from DiffCSP"
  * Each prompt's candidates first pass is_sensible (cell length 0.5-1000 A, angle 10-170 deg),
    then Structure.from_str
  * Each candidate must also pass is_valid = smact_validity (valence/electronegativity)
    AND structure_validity (min interatomic distance >= 0.5 A, volume >= 0.1)
  * For prompt i: min(matcher.get_rms_dist(pred, gt)[0]) over all candidates;
    no match at all -> None (unmatched)
  * match_rate = (#matched prompts) / (#prompts in the gen set)  -- unmatched samples stay in the denominator
  * RMSE definition (named mean_rms_dist in the original code): averaged over matched prompts only
      (what the user calls RMSE; the CrystaLLM code semantics are followed here, the two correspond
      one-to-one)
  * 1-shot = --num-gens 1 (take the 1st generation of each id = our smallest filename k);
    20-shot = --num-gens 20 (default behaviour of the official pipeline; any one matching candidate
    counts as a match)

Differences from the original file (the semantics above are unchanged):
  - Only the conditional generation (CSP) branch is ported; the unconditional branch
    (matminer fingerprints / COV / WDist) is not needed.
  - Input changed from tar.gz to a directory (same naming/structure: id__k.cif / id.cif),
    sorted by numeric k.
  - is_sensible inlined from crystallm/_metrics.py (regex body character-for-character identical).
  - Additionally outputs per-sample detail (matched / best rmsd / status of each candidate);
    the metric algorithm itself is unchanged.

Runtime environment: requires pymatgen + smact (on this machine the diffcsp / cdvae conda env:
pymatgen 2023.8.10 + smact 2.5.5; the official CrystaLLM requirements are pymatgen==2023.3.23,
smact==2.5.5, and the matcher/parsing APIs are equivalent between the two versions).
"""
import argparse
import itertools
import json
import os
import re
import warnings
from collections import Counter

import numpy as np
import smact
from smact.screening import pauling_test
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- validity helpers
# smact_validity / structure_validity / is_valid: copied function by function from
# CrystaLLM-main/bin/benchmark_metrics.py (that file notes it is adapted from
# DiffCSP scripts/eval_utils.py). Only its module-level matminer fingerprint
# dependency has been dropped.


def smact_validity(atom_types, use_pauling_test=True, include_alloys=True):
    # atom_types e.g. ["Fe", "Fe", "O", "O", "O"]
    elem_counter = Counter(atom_types)
    elems = [(elem, elem_counter[elem]) for elem in sorted(elem_counter.keys())]
    comp, elem_counts = list(zip(*elems))
    elem_counts = np.array(elem_counts)
    elem_counts = elem_counts / np.gcd.reduce(elem_counts)
    count = tuple(elem_counts.astype("int").tolist())

    elem_symbols = tuple(comp)
    space = smact.element_dictionary(elem_symbols)
    smact_elems = [e[1] for e in space.items()]
    electronegs = [e.pauling_eneg for e in smact_elems]
    ox_combos = [e.oxidation_states for e in smact_elems]
    if len(set(elem_symbols)) == 1:
        return True
    if include_alloys:
        is_metal_list = [elem_s in smact.metals for elem_s in elem_symbols]
        if all(is_metal_list):
            return True
    threshold = np.max(count)
    oxn = 1
    for oxc in ox_combos:
        oxn *= len(oxc)
    if oxn > 1e7:
        return False
    for ox_states in itertools.product(*ox_combos):
        stoichs = [(c,) for c in count]
        # Test for charge balance
        cn_e, cn_r = smact.neutral_ratios(
            ox_states, stoichs=stoichs, threshold=threshold)
        # Electronegativity test
        if cn_e:
            if use_pauling_test:
                try:
                    electroneg_OK = pauling_test(ox_states, electronegs)
                except TypeError:
                    # if no electronegativity data, assume it is okay
                    electroneg_OK = True
            else:
                electroneg_OK = True
            if electroneg_OK:
                return True
    return False


def structure_validity(crystal, cutoff=0.5):
    dist_mat = crystal.distance_matrix
    # Pad diagonal with a large number
    dist_mat = dist_mat + np.diag(
        np.ones(dist_mat.shape[0]) * (cutoff + 10.))
    if dist_mat.min() < cutoff or crystal.volume < 0.1:
        return False
    else:
        return True


def is_valid(struct):
    comp_valid = smact_validity(
        atom_types=[str(specie) for specie in struct.species]
    )
    struct_valid = structure_validity(struct)
    return comp_valid and struct_valid


# is_sensible: inlined from CrystaLLM crystallm/_metrics.py (body character-for-character identical).
_CELL_LENGTH_RE = re.compile(r"_cell_length_[abc]\s+([\d\.]+)")
_CELL_ANGLE_RE = re.compile(r"_cell_angle_(alpha|beta|gamma)\s+([\d\.]+)")


def is_sensible(cif_str, length_lo=0.5, length_hi=1000., angle_lo=10., angle_hi=170.):
    cell_lengths = _CELL_LENGTH_RE.findall(cif_str)
    for length_str in cell_lengths:
        length = float(length_str)
        if length < length_lo or length > length_hi:
            return False
    cell_angles = _CELL_ANGLE_RE.findall(cif_str)
    for _, angle_str in cell_angles:
        angle = float(angle_str)
        if angle < angle_lo or angle > angle_hi:
            return False
    return True


# ---------------------------------------------------------------- readers (dir-based)
def extract_cif_id(filepath):
    """Same as CrystaLLM extract_cif_id: the filename is assumed to be 'id__k.cif';
    returns the id (split on the last '__')."""
    filename = os.path.basename(filepath)
    parts = filename.rsplit("__", 1)
    if len(parts) == 2:
        id_part, _ = parts
        return id_part
    else:
        raise ValueError(f"'{filename}' does not conform to expected format 'id__n.cif'")


def read_generated_cifs(input_dir):
    """Directory version of read_generated_cifs:
    <input_dir>/<id>__<k>.cif -> {id: [cif text in ascending k order]}."""
    generated_cifs = {}
    for name in sorted(os.listdir(input_dir)):
        if not (name.endswith(".cif") and "__" in name):
            continue
        cif = open(os.path.join(input_dir, name), encoding="utf-8").read()
        cif_id = extract_cif_id(name)
        # ascending numeric k (corresponds to the write order in the CrystaLLM tar = generation order)
        k = int(name.rsplit("__", 1)[1][: -len(".cif")])
        if cif_id not in generated_cifs:
            generated_cifs[cif_id] = []
        generated_cifs[cif_id].append((k, cif))
    return {cid: [cif for _, cif in sorted(lst)] for cid, lst in generated_cifs.items()}


def read_true_cifs(input_dir):
    """Directory version of read_true_cifs: <input_dir>/<id>.cif -> {id: cif text}."""
    true_cifs = {}
    for name in sorted(os.listdir(input_dir)):
        if not name.endswith(".cif"):
            continue
        cif = open(os.path.join(input_dir, name), encoding="utf-8").read()
        cif_id = name[: -len(".cif")]
        true_cifs[cif_id] = cif
    return true_cifs


# ---------------------------------------------------------------- metrics (CSP branch)
def get_structs(id_to_gen_cifs, id_to_true_cifs, n_gens, length_lo, length_hi, angle_lo, angle_hi):
    """Same as CrystaLLM get_structs: per-id candidate list (after sensible + parse),
    one GT per id."""
    gen_structs = []
    true_structs = []
    for id, cifs in id_to_gen_cifs.items():
        if id not in id_to_true_cifs:
            raise Exception(f"could not find ID `{id}` in true CIFs")

        structs = []
        for cif in cifs[:n_gens]:
            try:
                if not is_sensible(cif, length_lo, length_hi, angle_lo, angle_hi):
                    continue
                structs.append(Structure.from_str(cif, fmt="cif"))
            except Exception:
                pass
        gen_structs.append(structs)

        true_structs.append(Structure.from_str(id_to_true_cifs[id], fmt="cif"))
    return gen_structs, true_structs


def get_match_rate_and_rms(gen_structs, true_structs, matcher):
    """Identical to CrystaLLM get_match_rate_and_rms (including the tqdm loop and
    exception handling).
    Returns (metrics dict, rms_dists list) -- rms_dists feeds the detail output,
    None = that prompt has no matching candidate.
    """
    def process_one(pred, gt, is_pred_valid):
        if not is_pred_valid:
            return None
        try:
            rms_dist = matcher.get_rms_dist(pred, gt)
            rms_dist = None if rms_dist is None else rms_dist[0]
            return rms_dist
        except Exception:
            return None

    rms_dists = []
    for i in range(len(gen_structs)):
        tmp_rms_dists = []
        for j in range(len(gen_structs[i])):
            try:
                struct_valid = is_valid(gen_structs[i][j])
                rmsd = process_one(gen_structs[i][j], true_structs[i], struct_valid)
                if rmsd is not None:
                    tmp_rms_dists.append(rmsd)
            except Exception:
                pass
        if len(tmp_rms_dists) == 0:
            rms_dists.append(None)
        else:
            rms_dists.append(np.min(tmp_rms_dists))

    rms_dists = np.array(rms_dists)
    match_rate = sum(rms_dists != None) / len(gen_structs)
    mean_rms_dist = rms_dists[rms_dists != None].mean()
    return {"match_rate": match_rate, "rms_dist": mean_rms_dist}, rms_dists


def detailed_per_candidate(gen_dir, ids_in_order, true_structs_by_id, matcher, n_gens,
                           length_lo, length_hi, angle_lo, angle_hi):
    """per-sample / per-candidate detail (computed the same way as get_match_rate_and_rms,
    for record-keeping)."""
    rows = []
    for pos, cid in enumerate(ids_in_order):
        cif_paths = sorted(
            [os.path.join(gen_dir, f) for f in os.listdir(gen_dir)
             if f.endswith(".cif") and f.startswith(cid + "__")],
            key=lambda p: int(os.path.basename(p).rsplit("__", 1)[1][: -len(".cif")]),
        )[:n_gens]
        gt = true_structs_by_id[cid]
        cand_rows = []
        for path in cif_paths:
            text = open(path, encoding="utf-8").read()
            rec = {"candidate": os.path.basename(path), "sensible": False,
                   "parse_ok": False, "valid": False, "rmsd": None}
            try:
                if not is_sensible(text, length_lo, length_hi, angle_lo, angle_hi):
                    cand_rows.append(rec)
                    continue
                rec["sensible"] = True
                pred = Structure.from_str(text, fmt="cif")
                rec["parse_ok"] = True
            except Exception:
                cand_rows.append(rec)
                continue
            try:
                rec["valid"] = is_valid(pred)
                rms_dist = None if not rec["valid"] else matcher.get_rms_dist(pred, gt)
                rec["rmsd"] = None if rms_dist is None else float(rms_dist[0])
            except Exception:
                pass
            cand_rows.append(rec)
        matched = [c for c in cand_rows if c["rmsd"] is not None]
        rows.append({
            "id": cid,
            "n_candidates": len(cif_paths),
            "n_sensible": sum(c["sensible"] for c in cand_rows),
            "n_parsed": sum(c["parse_ok"] for c in cand_rows),
            "n_valid": sum(c["valid"] for c in cand_rows),
            "n_matched": len(matched),
            "matched": len(matched) > 0,
            "best_rmsd": min(c["rmsd"] for c in matched) if matched else None,
            "candidates": cand_rows,
        })
    return rows


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="CSP benchmark: match rate and RMS distance "
                                             "(port of CrystaLLM benchmark_metrics.py, CSP branch).")
    ap.add_argument("gen_dir", help="directory: generated <id>__<k>.cif candidate files")
    ap.add_argument("gt_dir", help="directory: true <id>.cif files")
    ap.add_argument("--num-gens", type=int, default=0,
                    help="maximum number of candidates used per prompt. Default 0 = all. "
                         "1 = 1-shot (first generation), "
                         "20 = 20-shot (the official default --num-gens 20).")
    ap.add_argument("--length_lo", type=float, default=0.5)
    ap.add_argument("--length_hi", type=float, default=1000.)
    ap.add_argument("--angle_lo", type=float, default=10.)
    ap.add_argument("--angle_hi", type=float, default=170.)
    ap.add_argument("--out-json", default=None, help="summary JSON output path (contains metrics)")
    ap.add_argument("--detail-json", default=None, help="per-sample detail JSON output path")
    args = ap.parse_args()

    n_gens = args.num_gens
    if n_gens == 0:
        n_gens = None
        print("using all available generations...")
    else:
        print(f"using a maximum of {n_gens} generation(s) per compound...")

    # defaults taken from DiffCSP (identical to CrystaLLM benchmark_metrics.py)
    struct_matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)

    id_to_gen_cifs = read_generated_cifs(args.gen_dir)
    id_to_true_cifs = read_true_cifs(args.gt_dir)
    ids_in_order = list(id_to_gen_cifs.keys())

    gen_structs, true_structs = get_structs(
        id_to_gen_cifs, id_to_true_cifs, n_gens,
        args.length_lo, args.length_hi, args.angle_lo, args.angle_hi)
    metrics, rms_dists = get_match_rate_and_rms(gen_structs, true_structs, struct_matcher)
    print(metrics)

    out = {
        "n_prompts": len(ids_in_order),
        "num_gens": n_gens,
        "match_rate": float(metrics["match_rate"]),
        "mean_rmsd_matched": float(metrics["rms_dist"]),
        "n_matched": int(np.sum(rms_dists != None)),
    }
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"summary -> {args.out_json}")

    if args.detail_json:
        true_structs_by_id = {cid: s for cid, s in zip(ids_in_order, true_structs)}
        rows = detailed_per_candidate(args.gen_dir, ids_in_order, true_structs_by_id,
                                      struct_matcher, n_gens,
                                      args.length_lo, args.length_hi, args.angle_lo, args.angle_hi)
        with open(args.detail_json, "w") as f:
            json.dump({"summary": out, "samples": rows}, f, indent=1)
        print(f"detail -> {args.detail_json}")


if __name__ == "__main__":
    main()
