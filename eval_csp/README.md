# DMLM-CSP benchmark evaluation (Task B)

**1-shot and 20-shot** CSP performance (Match Rate / RMSE) of DMLM-CSP on
MP-20 / Carbon-24 / MPTS-52 / Perov-5. The evaluation logic is ported verbatim from
the CSP branch of `bin/benchmark_metrics.py` in a CrystaLLM checkout.
The CrystaLLM repository stays read-only; all scripts live in this directory.

## Definitions (A: where the protocol comes from)

| Item | Definition | Source |
|---|---|---|
| Evaluation unit | every id in a CrystaLLM prompt tar = one composition prompt; GT = that row's own original test CIF (the csv `cif` column; when the same composition appears as several structures, each one is a separate evaluation item) | `resources/benchmarks/<ds>/test.csv`, prompt tars |
| Match Rate | matched prompts / prompts-in-gen-set (unmatched ones stay in the denominator); matched = at least 1 candidate for that prompt yields an RMSD from the matcher | `benchmark_metrics.py::get_match_rate_and_rms` |
| RMSE definition | `mean_rms_dist` in the original code: the mean over matched prompts of the `min_candidate RMSD` | same (what the user calls RMSE) |
| matcher | `StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)` ("defaults taken from DiffCSP") | `benchmark_metrics.py` comment |
| Candidate validity chain | `is_sensible` (cell length 0.5-1000 A, angle 10-170 deg) -> `Structure.from_str` -> `is_valid` = smact_validity AND structure_validity (minimum interatomic distance >= 0.5 A, volume >= 0.1) | `benchmark_metrics.py` (from DiffCSP eval_utils) |
| 1-shot | `--num-gens 1`: take only the first generation of each id (k=0 in our filenames) | `benchmark_metrics.py --num-gens` |
| 20-shot | `--num-gens 20`: up to 20 independent candidates per id; any one match counts as matched | official pipeline default |
| GT and candidate format | GT `<id>.cif` (the csv text as-is); candidates `<id>__<k>.cif`, the id taken after the last `__`, ascending numeric k = generation order | `benchmark_metrics.py` tar convention |

> The paper's supplementary PDF could not be parsed, so per the user's instruction the
> CrystaLLM source on the server is the authoritative definition.

## Data and files (per dataset `data/<ds>/`)

| File | Content | Generating script |
|---|---|---|
| `orig/<id>.cif` | GT CIF (= the test.csv `cif` column, stripped + newline; all 22956 verified parseable with pymatgen 2023.8.10 `Structure.from_str`) | `prepare_test_data.py` |
| `test_input.pkl.gz` | `[(id,cif)]`, for read-only execution by the official preprocess | same |
| `budgets.json` | `{id: budget_tokens}`, deterministically fixed | `build_budgets.py` |
| `budget_meta.json` | per prompt comp/cond_len/number of train rows/budget basis + `length_basis` | same |

The prompt texts are in `prompts/<ds>/*.txt` (`data_<full-cell formula>\n`, matching the
decoding of the official prompt tars; none of the 4 datasets has bracketed groups).

### Length budget protocol (the only mechanism outside CrystaLLM; requires explanation)

DMLM is fixed-length diffusion and cannot stop on its own; the length budget makes the
candidate length match the model's prior for "how long the CIF of this composition is", in
order to emulate the stopping of autoregressive decoding:

```
budget = median(original token length of the train/val rows with the same reduced composition) - cond_len + 8
        , clamped to [50, max_total(=1005) - cond_len]
```

- Composition missing from the corpus (composition unique to test) -> dataset-level global
  median, recorded as `comp_miss`.
- Row boundaries: official `starts_<ds>_<split>.pkl`; carbon_24/mpts_52/perov_5 have no val
  starts, so "the position of the `data_` token = row start" is derived instead (first
  validated by asserting that the official train starts equal the derived ones; for mp_20 the
  9047 official starts equal the data_ positions one by one).
- Median figures (built 2026-09-08): mp_20 9046 prompts (6826 comp-miss, median budget 352,
  range 198-905); carbon_24 2030 (0 miss, 291-292, corpus comps=1, all C); mpts_52
  8095 (7248 miss, 4 clamped, median 366, 197-996); perov_5 3785 (836 miss, 319-436).

## Generation (generate_candidates.py)

```
python generate_candidates.py --ckpt <ds-ckpt> \
  --prompts-dir prompts/<ds> --budgets-json data/<ds>/budgets.json \
  --outdir data/<ds>/gen_20shot --shots 20 --max-iter 500 \
  --batch-size 8 --seed 1337 --max-total 1005
```

- condition = `data_<formula>\n` tokenized with the same tokenizer as the training rows
  (roundtrip assertion); the sequence = cond + budget [MASK]s;
  `dmlm.generate(max_iter=500, condition_mask, condition_ids=input_tokens)` (the same
  decoding path as generate_csp.py).
- Sampling randomness: per batch `torch.manual_seed(seed + global start row index)` -- tied to
  the batch split, so full runs fix batch-size=8 and record it.
- Filenames `{id}__{k}.cif`; k is globally consecutive (0..prompts x shots - 1), and ascending
  k within an id = generation order; decoding truncates at the first pad/eos/bos.

## Metrics (csp_metrics.py, diffcsp env: pymatgen 2023.8.10 + smact 2.5.5)

```
python csp_metrics.py data/<ds>/gen_20shot data/<ds>/orig \
  --num-gens 1|20 --out-json ... --detail-json ...
```

Output: `n_prompts / num_gens / match_rate / mean_rmsd_matched / n_matched` + per-sample
(n_sensible/parse/valid/matched, best_rmsd, per-candidate detail).

**Validation status (CPU synthetic test, 2026-09-08)**: 4 ids in `data/synth/` constructed as
20 copies of the GT (A), 20 candidates of a different material (B), 20 junk entries (C),
19 junk + 1 copy (D) -> 20-shot MR 0.5 / mean_rmsd ≈ 3.8e-16; 1-shot MR 0.25 -- exactly as
hand-computed (a copy matches at ≈0 RMS, a different material is judged unmatched but stays in
the denominator, junk is not counted as a candidate, 1-shot takes only k=0, and the detail
file agrees with the summary).

## Smoke results (mp_20, checkpoint step ~9836/45000 = 22%, 2026-09-09)

5 prompts x 20 shots = 100 candidates (run on GPU4 alongside training; seed 1337, batch 8,
max-iter 500) -> `data/mp_20/smoke_gen/`, metrics `smoke_summary_{1,20}.json` /
`smoke_detail_{1,20}.json`:

| Item | MR | mean_rmsd_matched | n_matched |
|---|---|---|---|
| 1-shot | 0.0 | — | 0/5 |
| 20-shot | 0.0 | — | 0/5 |

Stratification (20 candidates per prompt): sensible 19-20/20, parse 16-20/20, valid 0-20/20;
sameelems (the candidate's parsed element set ⊇ GT) 16-20/20; samecomp (identical reduced
formula) only 1/100 (mp-10009__9: GaTe but 6 atoms vs GT 8, vol 1219 vs 272.8).

Root-cause assessment (not a pipeline bug):
- The text level is self-consistent: the header of mp-1001034__0 is `data_Mg8In16Se32` (the cond
  text is protected, not model-generated), `_chemical_formula_sum` and the multiplicity
  declarations are mutually consistent with 8:16:32=1:2:4, and the atom loop terminates
  properly.
- The structure level disagrees: most candidates are complete CIFs of the "self-chosen Z /
  spinel-convention cell" kind (e.g. a=11.26 A vs GT 8.10 A), and pymatgen 2023.8.10 parses
  rows that carry a multiplicity but no space-group symbol as 1 site per row (4 sites
  Mg1In1Se2 vs GT 14 sites Mg2In4Se8) -> stoichiometry and lattice deviate from the GT -> the
  matcher returns None across the board.
- Conclusion: the checkpoint is only 22% trained (step ~10k/45k, val/ppl 1.27 vs best 1.218);
  it has learned element-set conditioning and the CIF text format but not yet exact
  stoichiometry / lattice reproduction -- MR=0 is the expected low level for that stage, not an
  error in the evaluation chain (the synthetic test hitting MR 0.5/0.25 exactly proves the
  chain).

Suggestion: once training has progressed, run the smoke test on the same 5 prompts again to
watch the MR trend (e.g. at step ~20k/30k/45k).

## Smoke results (mp_20, several checkpoints compared, 2026-09-09)

The same 5 prompts x 20 shots (cuda:3, seed 1337, batch 8, max-iter 500), diffcsp env metrics:

| checkpoint | training val/loss | 1-shot MR | 20-shot MR | n_matched | mean_rmsd |
|---|---|---|---|---|---|
| 0728 best.ckpt (best unconditional pretraining) | 0.082 (uncond protocol) | 0.0 | 0.0 | 0/5 | —(nan) |
| v1 step_999 (best existing conditional fine-tune) | 0.187 | 0.0 | 0.0 | 0/5 | —(nan) |
| 0728 last.ckpt (final unconditional pretraining state) | 0.254 (uncond protocol) | 0.0 | 0.0 | 0/5 | —(nan) |
| v1 step ~9836 (22% trained, morning smoke) | ppl 1.27 | 0.0 | 0.0 | 0/5 | —(nan) |

Assessment: the evaluation chain is validated (synthetic test 1-shot 0.25 / 20-shot 0.5 as
hand-computed), and the table above is real model output -- none of the four checkpoints
produced a single candidate that passes the matcher. Initial mechanistic reading: the distance
between the token-level objective and "exact unit-cell reproduction" is large (ppl 1.2/token x
~300 tokens -> probability of an exactly correct row ~e-60), so the trend must be re-measured
after conditional fine-tuning brings val down substantially. Generation directories
`data/mp_20/smoke_gen_{0728best,step999,0728last}/`.

## Protocol correction (2026-09-10): fixed-length GT-mask initialisation + official GT representation

The user pointed out that "the model is particularly sensitive to the initial sequence length;
at test time one should take the ground-truth test data, keep the composition information
unchanged and mask all remaining tokens, with the masked length equal to the unmasked length".
The generation-length protocol was re-examined accordingly and two root causes were found:

### Root cause 1: the official representation of GT/prompts is a "standardized CIF", not the raw cif of test.csv

Official `bin/prepare_csv_benchmark.py`: `csv cif -> Structure.from_str -> CifWriter(struct,
symprec=0.1)` (standardized writing) -> that text is the source of the official prompts and of
the **true cifs** (`preprocess.py` only moves `_chemical_formula_sum` onto the `data_` line).
The raw cif of test.csv is a **different cell** from it (fully expanded P1, often 1/2-1/4 of
the standardized cell), therefore:

- Of the 9046 prompts, only 3821 (42.1%) have a formula matching the csv
  `_chemical_formula_sum`; the rest are 2x (2829) / 3x (625) / 4x (1753) the csv cell, plus 29
  with inconsistent element ratios.
- Generation so far used the official prompt as condition (correct), but the **length budget
  was derived from csv statistics** and the **metric GT was the raw csv text** (the wrong
  representation) -> inconsistent with the site count/cell of the model's output, so the
  matcher was bound to fail.

### Root cause 2: the length protocol (now replaced per the user's protocol)

New protocol (`generate_gtmask.py`):
```
initial value = token sequence of the GT canonical row (output of prepare_gt.py)
       (official chain: csv -> CifWriter(symprec=0.1) -> the four preprocess.py steps ->
         tokenize_cifs.preprocess row cleaning -> CIFTokenizer, no eos)
composition = the tokens of the first line data_<formula>\n -- left unchanged
              (protected by partial_mask/condition_mask)
the rest = all [MASK]; total length = the GT row's original token count
           (the same length before and after masking)
```

### Validation (2026-09-10, all scripts in eval_csp/)

| Check | Result |
|---|---|
| canonical first line of the 9046 test ids vs the official prompt tars | **9046/9046 character-for-character identical** (0 mismatches, 0 unk, 0 failures) |
| GT canonical row length (test, 9046) | mean 374.7 / med 352 / p10 262 / p90 487 |
| corpus val.bin row length (9047, the model's real training distribution) | mean 375.2 / med 352 / p10 262 / p90 509 |
| val.csv -> official chain -> canonical text vs val.bin text | character length mean 901.4/med 875 vs 898.4/872; the content differs line by line only in spaces (the corpus keeps two-space alignment, ours uses one space; the tokenizer collapses spaces => the token count is unchanged) |
| deviation of the old-protocol budgets | 5 smoke ids: GT row length 260/259/331/333/293 vs budget 395/354/350/357/263 (the mask region is up to +135 tokens too long) |

Scripts: `prepare_gt.py` (builds the GT rows + `data/<ds>/gt_prep/<id>.cif` + `gt_rows.json` +
`gt_rows_tokens.pkl.gz`); `generate_gtmask.py` (generation under the new protocol,
`--ckpt/--ds/--ids-json/--outdir/--shots/--device`); `validate_prep_prepared.py` (val-side
comparison).

## Status (2026-09-10 01:30)

- v2 (init v1 step_999, lr 1e-4) val degraded monotonically 0.19 -> 0.30 plateau (~step 4500;
  best.ckpt never surpassed the init) -> stopped at 01:28, checkpoint kept in
  `logs/dmlm_150m_csp_mp20_v2/checkpoints/`.
- v3 started (01:30): init=0728 best.ckpt (healthy uncond plateau at 0.082), lr 1e-4 -> 1e-5
  @20000 steps, a single-variable experiment: does a healthy init let cond val decrease
  normally (criterion: the first 4-6 val points must not be higher than the first val).
- The other 3 datasets' checkpoints are untrained; the data side
  (prompts/orig/budgets/metrics/generation scripts) is fully ready.

## Re-test results (2026-09-10 ~02:20): 0728 best.ckpt, new protocol + official postprocess

Generation: `generate_gtmask.py` (fixed-length GT-mask initialisation: composition tokens
frozen, everything else [MASK], total length = the GT row's token count) ->
`data/mp_20/gen_gtmask_0728best/` (5 ids x 20 shots = 100 candidates, cuda:3, seed 1337,
max-iter 500, batch 8, run alongside v3 training, ~18 min).
Postprocessing: `postprocess_candidates.py` (verbatim port of the official `bin/postprocess.py`:
`extract_space_group_symbol` -> `replace_symmetry_operators` -> `remove_atom_props_block`) ->
`gen_gtmask_0728best_post/`.
Metrics: `csp_metrics.py` (diffcsp env), GT from `data/mp_20/gt_prep/` (the official true-cifs
representation).

| Candidate set | GT | 1-shot MR | 1-shot mean_rmsd | 20-shot MR | 20-shot mean_rmsd |
|---|---|---|---|---|---|
| New protocol + official postprocess | gt_prep | **0.6** (3/5) | 0.1774 | **0.8** (4/5) | 0.0915 |
| New protocol + official postprocess | orig (raw csv) | 0.6 | 0.1774 | 0.8 | 0.0915 |
| New protocol, not postprocessed | gt_prep | 0.0 | — | 0.0 | — |
| Old protocol (budget-length mask), postprocessed | gt_prep / orig | 0.0 | — | 0.0 | — |

per-id (20-shot): mp-1001012 0.0057 ✓ / mp-1001034 0.0066 ✓ / mp-10014 0.0202 ✓ /
mp-10009 0.3336 ✓ / mp-1001 ✗ (all 20 candidates unmatched).

**Conclusions (attribution for this round's changes)**:
1. **The fixed-length GT-mask initial value is the decisive factor**: same checkpoint, same
   5 ids, same official postprocess -- the old protocol (budget mask from the corpus median
   length) gives MR=0.0, the new protocol MR=0.6/0.8. Even postprocessed, all old candidates
   are unmatched -- the condition-distribution shift caused by the wrong length is the main
   factor (consistent with the user's "the model is sensitive to the initial sequence length").
2. **The official postprocess is a necessary condition**: without postprocess the new-protocol
   candidates give MR=0.0 (the corpus style lists only `1 'x, y, z'` plus the site multiplicity,
   so pymatgen expands via the listed operations -> 2 sites vs GT 8 sites, and the matcher must
   fail); after postprocess the same candidates give MR=0.6/0.8.
3. Using gt_prep vs the raw csv text as GT gives identical metrics on these 5 ids (difference
   <1e-6); gt_prep is taken as authoritative (the official representation).
4. All candidates are sensible 20/20; valid (smact AND structure) 10-20/20; the 20-shot gain
   comes mainly from mp-1001034 (only k>0 matches).

Scripts/artefacts: `postprocess_candidates.py`, `gen_gtmask_0728best.log`,
`data/mp_20/m_{raw|post}_vs_{gtprep|orig}_{1,20}.json`, `data/mp_20/md_post_vs_gtprep_*.json`,
`data/mp_20/m_{oldpost}_{gt_prep|orig}_{1,20}.json`.
