# DMLM — Diffusion Language Model for Crystal Structure Generation

A masked-diffusion language model over CIF token sequences. The model denoises a
sequence of CIF tokens in a fixed number of steps instead of generating it
left-to-right, which makes it suitable both for **unconditional** crystal
generation and for **composition-conditioned crystal structure prediction (CSP)**,
where the composition is pinned and the structural part is denoised.

The backbone is ESM2-t30 (150M) with the protein tokenizer replaced by a CIF
tokenizer; it is trained from scratch, no pretrained protein weights are used.

## Repository layout

```
train.py                     training entry point (Hydra)
test.py                      validation / test / predict entry point (Hydra)
generate_csp.py              composition-conditioned sampling from a checkpoint
sanity_checks_csp.py         self-contained assertion suite (no pytest needed)

configs/                     Hydra configuration
  config.yaml                  root config
  config_all_150m_mp20.yaml    model config loaded by DMLM.from_pretrained()
  datamodule/crystalmols_hf.yaml
  experiment/dmlm/             training recipes (base, CSP, MP-20 75% / 50% subsets)
  paths/default.yaml           output paths, driven by ${oc.env:PROJECT_ROOT}
  hydra/, trainer/, callbacks/

src/byprot/                  the library
  models/dmlm/dmlm.py            the diffusion model (forward, loss, sampling)
  models/dplm/modules/           ESM2 backbone with the CIF embedding hook
  datamodules/crystalmols_hf.py  CrystalDataset / CrystalMolsDataModule
  crystallm/                     CIF tokenizer and CIF text utilities
  tokenizers/                    vocab wrapper over meta.pkl
  tasks/lm/dmlm.py               LightningModule
  training_pipeline.py, testing_pipeline.py
  utils/                         config, callbacks, optim, schedulers, registry

data-bin/crystalmols/        tokenized datasets (see below)
eval_csp/                    CSP benchmark: generation, postprocess, metrics
```

## Installation

Python 3.9 is what the pinned stack was tested with.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The CSP evaluation in `eval_csp/` has its own dependency set (pymatgen, smact)
and runs in a separate environment, because those packages want a different
numpy/pymatgen stack than the training code:

```bash
python -m venv .venv-eval && source .venv-eval/bin/activate
pip install -r requirements-eval.txt
```

Nothing needs to be `pip install`ed for the library itself: `train.py`,
`test.py`, `generate_csp.py` and `sanity_checks_csp.py` all put `src/` on
`sys.path` themselves.

Optional machine-specific settings (paths, eval interpreter) go in a `.env` file
at the repository root — copy `.env.example`.

The ESM2 backbone configuration is fetched from the Hugging Face Hub on first
use and cached under `~/.cache/huggingface`. On a machine without outbound
network access, warm the cache once and then run offline:

```bash
HF_HUB_OFFLINE=1 python sanity_checks_csp.py
```

## Datasets

All tokenized data lives under `data-bin/crystalmols/`. Every dataset directory
follows the same format, produced by tokenizing CIF text with the CIF tokenizer:

| file | contents |
|---|---|
| `train.bin`, `val.bin` | flat `np.uint16` arrays of token ids, one concatenated stream |
| `starts_<name>_<split>.pkl` | token offset at which each CIF row starts |
| `meta.pkl` | vocabulary (`stoi` / `itos`), 371 CIF tokens + specials |

A row is not length-prefixed; the row boundaries are the entries of the
corresponding `starts` file. If a `starts` file is absent the loader recovers the
boundaries by splitting on the `data_` token id, which begins every CIF row.
`extract_starts.py` regenerates the `starts` files from a `.bin`.

| directory | rows (train / val) | notes |
|---|---|---|
| `data-bin/crystalmols/` | 2,047,889 / 227,544 | full corpus (v1), `starts_v1_*.pkl` |
| `tokens_mp_20/` | 27,136 / 9,047 | MP-20 |
| `tokens_mp_20_75/` | 20,516 / 6,848 | 75% subset, with `index_map_*.npy` into MP-20 and a `subset_manifest.json` |
| `tokens_mp_20_50/` | 13,778 / 4,693 | 50% subset, same layout |
| `tokens_carbon_24/` | 6,091 / — | val split recovered by auto-split |
| `tokens_mpts_52/` | 27,377 / — | val split recovered by auto-split |
| `tokens_perov_5/` | 11,356 / — | val split recovered by auto-split |

The 75% and 50% subsets are index selections of MP-20; `index_map_train.npy` and
`index_map_val.npy` record which MP-20 row each subset row came from, and
`build_report.txt` records how they were built.

### Data path configuration

`configs/paths/default.yaml` resolves the root from the `PROJECT_ROOT`
environment variable, which `train.py` / `test.py` set to their own directory.
Override it in `.env` to keep data or logs on another disk:

```
PROJECT_ROOT=/path/to/repo
```

Which dataset a run uses is set by `datamodule.data_dir` in the experiment
config, e.g. `configs/experiment/dmlm/dmlm_mp20_75.yaml`:

```yaml
datamodule:
  data_dir: ${paths.data_dir}/tokens_mp_20_75
  vocab_file: ${paths.data_dir}/tokens_mp_20_75/meta.pkl
```

`${paths.data_dir}` is `${paths.root_dir}/data-bin/crystalmols`.

## Training

```bash
# composition-conditioned training (CSP)
python train.py experiment=dmlm/dmlm_base_csp

# unconditional training
python train.py experiment=dmlm/dmlm_base

# MP-20 75% / 50% subsets
python train.py experiment=dmlm/dmlm_mp20_75
python train.py experiment=dmlm/dmlm_mp20_50
```

Any config value can be overridden on the command line:

```bash
python train.py experiment=dmlm/dmlm_base_csp \
    datamodule.batch_size=32 trainer.accumulate_grad_batches=40 \
    trainer.devices=[0,1,2,3] train.lr=1e-4 trainer.max_steps=20000
```

Useful knobs, all in `configs/config.yaml` and the experiment files:

- `datamodule.conditioning` — `composition` for CSP, `none` for unconditional.
- `trainer.max_steps`, `trainer.val_check_interval` — the training length and how
  often validation runs. `configs/experiment/dmlm/dmlm_base_csp.yaml` raises
  `val_check_interval` to 2500 because a CSP validation pass is expensive.
- `train.ckpt_path` — `last.ckpt` **resumes** a run (optimizer state, scheduler
  and `global_step` are all restored).
- `train.init_ckpt` — loads **weights only**; optimizer state and `global_step`
  restart from zero. This is not a resume, and mixing the two up silently resets
  the schedule.

Logs and checkpoints are written to `${paths.log_dir}/${name}`, i.e.
`logs/<run name>/` by default.

## Validation, test and prediction

`test.py` uses `configs/test.yaml` and needs an experiment directory (the Hydra
output directory of a past run) plus a checkpoint:

```bash
python test.py experiment_path=logs/dmlm_150m_csp \
    ckpt_path=/path/to/checkpoint.ckpt data_split=val

# `mode=predict` runs the prediction branch instead of the test branch
python test.py experiment_path=logs/dmlm_150m_csp \
    ckpt_path=/path/to/checkpoint.ckpt data_split=test mode=predict
```

### Checkpoints

No model weights are shipped with this repository. Train one, or supply your own
`*.ckpt` from a Lightning run — both `test.py --ckpt_path` and
`generate_csp.py --checkpoint` take a path to one. The model config is stored
inside the checkpoint (`DMLM.from_pretrained` reads it back), so a checkpoint
trained from a different experiment config still loads.

## Generation

### Composition-conditioned (CSP)

```bash
# 10 candidate structures for one composition
python generate_csp.py --checkpoint /path/to/last.ckpt \
    --composition NaCl --num-samples 10 --outdir csp_output

# several compositions, 5 candidates each
python generate_csp.py --checkpoint /path/to/last.ckpt \
    --composition NaCl --composition SiO2 --composition LiFePO4 \
    --num-samples 5 --device cuda:0
```

Writes `<outdir>/<composition>/candidate_<i>.cif`. The composition prefix
(`data_<formula>\n`) is pinned before sampling and force-restored after every
denoising step, so it cannot drift; the script asserts this and aborts otherwise.
`--seq-len` sets the total sequence length and `--max-iter` the number of
denoising steps.

### Unconditional

There is no dedicated CLI for unconditional sampling; call the model directly.
This is the same call the sanity suite exercises:

```python
import sys; sys.path.insert(0, "src")
import torch
from byprot.models.dmlm.dmlm import DiffusionMaterialLanguageModel as DMLM

model = DMLM.from_pretrained("/path/to/last.ckpt").eval().cuda()
input_tokens = torch.full((4, 500), model.mask_id, dtype=torch.long, device="cuda")
with torch.no_grad():
    samples = model.generate(input_tokens=input_tokens, max_iter=500,
                             sampling_strategy="argmax")
print([model.tokenizer.decode(s.tolist()) for s in samples])
```

## CSP benchmark evaluation

`eval_csp/` reproduces the standard 1-shot / 20-shot match-rate protocol. It is
self-contained and expects its own environment (see Installation).

The protocol has three mandatory steps; a candidate that skips the postprocess
scores a match rate of zero:

1. `generate_gtmask.py` — build a fixed-length initialisation from the ground-truth
   row: keep the composition tokens, set everything else to `[MASK]`, so the
   sequence length equals the true structure's length.
2. `postprocess_candidates.py` — the official postprocess.
3. `csp_metrics.py` — `StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)` plus the
   official validity chain (`is_sensible` → `Structure.from_str` → smact validity
   and minimum interatomic distance).

All of it is driven by one script:

```bash
DMLM_PY=/path/to/train-env/bin/python \
EVAL_PY=/path/to/eval-env/bin/python \
./eval_csp/run_eval_csp.sh --ckpt /path/to/last.ckpt --name myrun --device cuda:3
```

Useful flags: `--shots`, `--max-iter`, `--limit` (quick smoke run), `--ds`
(dataset), `--gt-sets`, `--num-gens`. Metrics are written next to the data as
`eval_csp/data/<ds>/m20_*.json` and summarised in `eval_csp/eval_summary.txt`.

`eval_csp/data/mp_20/` ships with the ground truth (`orig/`, `gt_prep/`,
`gt_rows.json`, `gt_rows_tokens.pkl.gz`, `test_input.pkl.gz`) and the 9,046
benchmark prompts in `eval_csp/prompts/mp_20/`, so evaluation can run without
regenerating anything. The data-preparation scripts (`prepare_test_data.py`,
`prepare_gt.py`, `preprocess_test.py`, `build_budgets.py`) rebuild those files
and read the third-party benchmark CSVs from a CrystaLLM checkout pointed at by
`CRYSTALLM_ROOT` — see `.env.example`.

## Sanity checks

```bash
python sanity_checks_csp.py                  # auto-detects the device
SANITY_DEVICE=cpu python sanity_checks_csp.py
```

Thirteen assertions covering the tokenizer and special-token ids, corruption
invariance across timesteps, loss masking, sampling initialisation, and that a
batch of real tokenized CIFs loads correctly. Exits non-zero on failure.

## License and attribution

Apache-2.0, see `LICENSE`. The library under `src/byprot/` derives from
[ByProt](https://github.com/bytedance/ByProt) (Copyright (c) 2024 Bytedance Ltd.
and/or its affiliates); the CIF tokenizer and CIF text utilities under
`src/byprot/crystallm/` derive from [CrystaLLM](https://github.com/lantunes/CrystaLLM).

## Before publishing this repository

- `data-bin/crystalmols/train.bin` is 1.4 GB and `val.bin` is 149 MB. Both exceed
  GitHub's 100 MB per-file limit, so the repository cannot be pushed to GitHub as
  it stands. Use Git LFS, host the data separately, or drop these two files if the
  per-dataset directories under `data-bin/crystalmols/` are sufficient.
- The datasets are redistributed here; confirm that their upstream licenses permit
  it. `eval_csp/data/` and `eval_csp/prompts/` contain MP-20 benchmark data from
  CrystaLLM, and the benchmark CSVs themselves are read from a CrystaLLM checkout
  rather than vendored.
